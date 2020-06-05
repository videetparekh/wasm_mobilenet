const path = require("path");
const fs = require("fs");
const express = require("express");
const now = require("performance-now");

// Local Libraries
var models = require('../utils/model.js')
var preproc = require('../utils/preprocessor.js')
var debug = require('../utils/debug.js')
var utils = require('../utils/common_utils.js')
var runtime = require('../utils/runtime.js')


const router = express.Router();

router.get("/", (req, res) => {
    res.render("predict");
});

// Display the dashboard page
router.post("/", async(req, res) => {
    var imageURI = req.body.image
    var modelType = req.body.modelType
    var variantType = req.body.variantType
    var modelInfo = models.collectModel(modelType, variantType)
    var labels = await utils.getLabels(modelInfo["labels"])

    var [classifier, loadtime] = await runtime.tvmSetup(modelInfo)
    var each = []
    const iters = 1 // 1

    for (let i = 0; i < iters; i++) {
        inf_start = now()
        var preprocstart = now()
        var imageData = preproc.drawCanvas(imageURI, modelInfo['input_shape'])
        var processedImage = modelInfo["preprocessor"](imageData)
        var preproctime = now() - preprocstart
        fs.writeFile("debug_js.txt", processedImage.join(), function(err) { if (err) { console.log(err) } })

        var label = await classifier.classify(processedImage)
        inf_end = now()
        inftime = inf_end - inf_start

        resp_obj = { "label": labels[label], "loadtime": loadtime.reduce((a, b) => a + b, 0), "inftime": inftime }
        resp_obj["read_wasm"] = loadtime[0]
        resp_obj["load_weights"] = loadtime[1]
        resp_obj["pop_weights"] = loadtime[2]
        resp_obj["preprocess"] = preproctime
            // console.log(resp_obj)
        each.push(resp_obj)
    }
    res.send(JSON.stringify(resp_obj));

    debug.writeToFile(each, ['label', 'loadtime', 'inftime', 'read_wasm', 'load_weights', 'pop_weights', "preprocess"], 'results/' + modelType + "/" + variantType + '.csv')
});

module.exports = router;