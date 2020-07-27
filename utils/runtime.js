const path = require("path");
const fs = require("fs");
const express = require("express");
const now = require("performance-now");

module.exports = {
    lreSetup: async function(modelInfo) {

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

        // LRE Loader WASM and create Runtime
        start = now()
        const lrejs = require("../runtime_dist");
        const wasmPath = lrejs.wasmPath();
        delete require.cache[require.resolve(path.join(wasmPath, "lrejs_runtime.wasi.js"))]
        const EmccWASI = require(path.join(wasmPath, "lrejs_runtime.wasi.js"));
        var WasiObj = new EmccWASI()
        if (modelInfo["input_type"] == "uint8" || modelInfo["input_type"] == "int8") {
            WasiObj['Module']['wasmLibraryProvider']['imports']['env']['roundf'] = Math.round
        }
        const wasmSource = fs.readFileSync(modelInfo["base"] + modelInfo["wasm"])
        console.log(wasmSource)
        const lre = await lrejs.instantiate(wasmSource, WasiObj)

        ctx = lre.cpu(0)
        const sysLib = lre.systemLib()
            // console.log(sysLib)
        const executor = lre.createGraphRuntime(graphJson, sysLib, ctx)
        loadtime[0] = now() - start

        // Populate weights into graph
        start = now()
        executor.loadParams(paramsBinary)
        const inputData = lre.empty(modelInfo["input_shape"], modelInfo["input_type"], lre.cpu());
        const outputData = lre.empty(modelInfo["output_shape"], modelInfo["input_type"], lre.cpu());
        const outputGPU = executor.getOutput(0);

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