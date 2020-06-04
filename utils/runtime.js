const path = require("path");
const fs = require("fs");
const express = require("express");
const now = require("performance-now");

module.exports = {
    tvmSetup: async function(modelInfo) {

        var loadtime = [0.0, 0.0, 0.0]

        //Collect and load weights and graph file
        var start = now()
        var temp = await JSON.parse(await fs.readFileSync(modelInfo["base"] + modelInfo["graph"], "utf-8"));
        delete temp['leip']
        const graphJson = JSON.stringify(temp)
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
        var WasiObj = new EmccWASI()
        if (modelInfo["input_type"] == "uint8" || modelInfo["input_type"] == "int8") {
            WasiObj['Module']['wasmLibraryProvider']['imports']['env']['roundf'] = Math.round
        }
        const wasmSource = fs.readFileSync(modelInfo["base"] + modelInfo["wasm"])
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
            return sortedIndex[0];
        }
        loadtime[2] = now() - start
        return [classifier, loadtime];
    }
}