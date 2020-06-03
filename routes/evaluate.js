const path = require("path");
const fs = require("fs");
const express = require("express");
const now = require("performance-now");

// Local Libraries
var modelLib = require('../utils/model.js')
var preProcLib = require('../utils/preprocessor.js')
var debugLib = require('../utils/debug.js')


const router = express.Router();

router.get("/", (req, res) => {
    res.render("evaluate");
});

// Display the dashboard page
router.post("/", async(req, res) => {
    var imageURI = req.body.image
    var modelType = req.body.modelType
    var variantType = req.body.variantType
    var modelInfo = modelLib.collectModel(modelType, variantType)

    var each = []
    const iters = 50 // 1

    for (let i = 0; i < iters; i++) {
        var [classifier, loadtime] = await tvm_setup(modelInfo)
        inf_start = now()
        var preprocstart = now()
        var imageData = preProcLib.drawCanvas(imageURI, modelInfo['input_shape'])
        var processedImage = modelInfo["preprocessor"](imageData)
        var preproctime = now() - preprocstart
        var label = await classifier.classify(processedImage)
        inf_end = now()
        inftime = inf_end - inf_start

        resp_obj = { "label": label, "loadtime": loadtime.reduce((a, b) => a + b, 0), "inftime": inftime }
        resp_obj["read_wasm"] = loadtime[0]
        resp_obj["load_weights"] = loadtime[1]
        resp_obj["pop_weights"] = loadtime[2]
        resp_obj["preprocess"] = preproctime
            // console.log(resp_obj)
        each.push(resp_obj)
    }
    res.send(JSON.stringify(resp_obj));

    debugLib.writeToFile(
        each, ['label', 'loadtime', 'inftime', 'read_wasm', 'load_weights', 'pop_weights', "preprocess"],
        'results/' + modelType + "/" + variantType + '.csv'
    )
});

async function tvm_setup(modelInfo) {

    var loadtime = [0.0, 0.0, 0.0]

    //Collect and load weights and graph file
    var start = now()
    var temp = await JSON.parse(await fs.readFileSync(modelInfo["base"] + modelInfo["graph"], "utf-8"));
    delete temp['leip']
    const graphJson = JSON.stringify(temp)
    const synset = await JSON.parse(await fs.readFileSync(modelInfo["labels"], ));
    const paramsBinary = new Uint8Array(
        await fs.readFileSync(modelInfo["base"] + modelInfo["params"])
    );
    loadtime[1] = now() - start

    // TVM Loader WASM and create Runtime
    start = now()
    const tvmjs = require("../runtime_dist");
    const wasmPath = tvmjs.wasmPath();
    delete require.cache[require.resolve(path.join(wasmPath, "tvmjs_runtime.wasi.js"))]
    const EmccWASI = require(path.join(wasmPath, "tvmjs_runtime.wasi.js"));
    // Try adding roundf to EMCC WASI here
    var WasiObj = new EmccWASI()
    WasiObj['Module']['wasmLibraryProvider']['imports']['env']['roundf'] = Math.round
    const wasmSource = fs.readFileSync(modelInfo["base"] + modelInfo["wasm"])

    //Lorenz Fails Here: LinkError: WebAssembly.instantiate(): Import #0 module="env" function="roundf" error: function import requires a callable
    const tvm = await tvmjs.instantiate(wasmSource, WasiObj)

    ctx = tvm.cpu(0)
    const sysLib = tvm.systemLib()
        // console.log(sysLib)
    const executor = tvm.createGraphRuntime(graphJson, sysLib, ctx)
    loadtime[0] = now() - start

    // Populate weights into graph
    start = now()
    executor.loadParams(paramsBinary)
    const inputData = tvm.empty(modelInfo["input_shape"], modelInfo["input_type"], tvm.cpu());
    const outputData = tvm.empty(modelInfo["output_shape"], modelInfo["input_type"], tvm.cpu());
    const outputGPU = executor.getOutput(0);
    // run the first time to make sure all weights are populated.
    // executor.run();
    // await ctx.sync();
    var end = now()

    classifier = {}

    classifier.classify = async(imageData) => {
        inputData.copyFrom(imageData);
        // console.log(inputData)
        executor.setInput(modelInfo["input_name"], inputData);
        executor.run();
        outputData.copyFrom(outputGPU);
        const sortedIndex = Array.from(outputData.toArray())
            .map((value, index) => [value, index])
            .sort(([a], [b]) => b - a)
            .map(([, index]) => index);
        // for (let i = 0; i < 5; ++i) {
        //     console.log("Top-" + (i + 1) + " " + synset[sortedIndex[i]]);
        // }
        return synset[sortedIndex[0]];
    }
    loadtime[2] = now() - start
    return [classifier, loadtime];
}

module.exports = router;