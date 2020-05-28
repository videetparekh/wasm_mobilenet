const path = require("path");
const fs = require("fs");
const express = require("express");
const assert = require("assert");
const fetch = require("fetch");
const now = require("performance-now");
const Canvas = require("canvas")
var stringify = require('csv-stringify');
var nj = require('numjs')

const router = express.Router();

// Display the dashboard page
router.post("/", async(req, res) => {
    var imageURI = req.body.image
    var modelType = req.body.modelType

    var each = []
    const iters = 1

    for (let i = 0; i < iters; i++) {
        var [classifier, loadtime] = await tvm_setup(modelType)
        inf_start = now()
        var imageData = drawCanvas(imageURI)
        var processedImage = preprocess_imagenet(imageData)
            // var processedImage = preprocImage(imageData)
            // var [label, loadtime, inftime]= await classify(processedImage);
        var label = await classifier.classify(processedImage)
        inf_end = now()
        inftime = inf_end - inf_start

        resp_obj = { "label": label, "loadtime": loadtime.reduce((a, b) => a + b, 0), "inftime": inftime }
        resp_obj["read_wasm"] = loadtime[0]
        resp_obj["load_weights"] = loadtime[1]
        resp_obj["pop_weights"] = loadtime[2]
            // console.log(JSON.stringify(resp_obj))
        each.push(resp_obj)
    }
    res.send(JSON.stringify(resp_obj));

    console.log(each)
        // writeToFile(each, ['label', 'loadtime', 'inftime', 'read_wasm', 'load_weights', 'pop_weights'])
});

async function tvm_setup(modelType) {

    var modelPaths = collectModel(modelType)

    var loadtime = [0.0, 0.0, 0.0]

    //Collect and load weights and graph file
    var start = now()
    var temp = await JSON.parse(await fs.readFileSync(modelPaths["base"] + modelPaths["graph"], "utf-8"));
    delete temp['leip']
    const graphJson = JSON.stringify(temp)
    const synset = await JSON.parse(await fs.readFileSync("public/wasm/open_images_labels.json", ));
    const paramsBinary = new Uint8Array(
        await fs.readFileSync(modelPaths["base"] + modelPaths["params"])
    );
    loadtime[1] = now() - start

    // TVM Loader WASM and create Runtime
    start = now()
    const tvmjs = require("../runtime_dist");
    const wasmPath = tvmjs.wasmPath();
    delete require.cache[require.resolve(path.join(wasmPath, "tvmjs_runtime.wasi.js"))]
    const EmccWASI = require(path.join(wasmPath, "tvmjs_runtime.wasi.js"));
    const wasmSource = fs.readFileSync(modelPaths["base"] + modelPaths["wasm"])
    const tvm = await tvmjs.instantiate(wasmSource, new EmccWASI())

    ctx = tvm.cpu(0)
    const sysLib = tvm.systemLib()
        // console.log(sysLib)
    const executor = tvm.createGraphRuntime(graphJson, sysLib, ctx)
    loadtime[0] = now() - start

    // Populate weights into graph
    start = now()
    const test = executor.loadParams(paramsBinary)
    const inputData = tvm.empty([1, 3, 224, 224], "float32", tvm.cpu());
    const outputData = tvm.empty([1, 10], "float32", tvm.cpu());
    const outputGPU = executor.getOutput(0);
    // run the first time to make sure all weights are populated.
    executor.run();
    await ctx.sync();
    var end = now()

    classifier = {}

    classifier.classify = async(imageData) => {
        inputData.copyFrom(imageData);
        // console.log(inputData)
        executor.setInput("input_1", inputData);
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

function preprocess_imagenet(imageData) {
    var rgbU8 = cleanAndStripAlpha(imageData)
    const numpyArray = nj.float32(rgbU8).reshape(1, 3, 224, 224)
        // var fp32data = nj.float32(numpyArray)
    var divfp32data = numpyArray.divide(127.5)
    var normdata = divfp32data.subtract(1.0)
    console.log(normdata.flatten().tolist())
    return new Float32Array(normdata.flatten().tolist());
}

function preprocess_uint8(imageData) {
    var rgbU8 = cleanAndStripAlpha(imageData)
    return rgbU8;
}

function cleanAndStripAlpha(imageData) {
    const width = imageData.width;
    const height = imageData.height;
    const npixels = width * height;

    const rgbaU8 = imageData.data;

    // Drop alpha channel. Resnet does not need it.
    const rgbU8 = new Uint8Array(npixels * 3);
    for (let i = 0; i < npixels; ++i) {
        rgbU8[i * 3] = rgbaU8[i * 4];
        rgbU8[i * 3 + 1] = rgbaU8[i * 4 + 1];
        rgbU8[i * 3 + 2] = rgbaU8[i * 4 + 2];
    }
    return rgbU8;
}

function drawCanvas(url) {
    var cvs = Canvas.createCanvas(224, 224);
    var context = cvs.getContext('2d');
    var img = new Canvas.Image;
    img.onload = function() {
        context.drawImage(img, 0, 0);
    };
    img.src = url;
    return context.getImageData(0, 0, 224, 224)
}

function collectModel(modelType) {

    // The base path is the path to the folder containing the params and graph files
    // The modelObj provides filenames relative to the basepath collected from ModelInfo
    const modelInfo = {
        "baseline": "public/wasm/baseline/",
        "reimann": "public/wasm/reimann/",
        "lorenz": "public/wasm/lorenz"
    }

    var modelObj = {
        "graph": "modelDescription.json",
        "params": "modelParams.params",
        "wasm": "modelLibrary.wasm"
    }

    modelObj["base"] = modelInfo[modelType]
    return modelObj
}


function writeToFile(data, column_headers) {
    stringify(data, { header: true, columns: column_headers }, function(err, output) {
        fs.writeFile('results.csv', output, 'utf8', function(err) {
            if (err) {
                console.log('Some error occured - file either not saved or corrupted file saved.');
            } else {
                console.log('It\'s saved!');
            }
        });
    });
}

module.exports = router;