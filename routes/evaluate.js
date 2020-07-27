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
    res.render("evaluate", models.getModelVariantList());
});

// Display the dashboard page
router.post("/", async(req, res) => {

    var data = req.body.data

    var modelType = req.body.modelType
    var variantType = req.body.variantType
    var modelInfo = models.collectModel(modelType, variantType)
    var labels = await utils.getLabels(modelInfo["labels"])

    var predData = []
    var [classifier, loadtime] = await runtime.lreSetup(modelInfo)
    var total_start = now()
    for (let i = 0; i < data.length; i++) {
        inf_start = now()
        var preprocstart = now()
        var imageData = preproc.drawCanvas(data[i].uri, modelInfo['input_shape'])
        var processedImage = modelInfo["preprocessor"](imageData)
        var preproctime = now() - preprocstart
        var predIndex = await classifier.classify(processedImage)
        inftime = now() - inf_start

        resp_obj = {
            "predIndex": predIndex,
            "originalIndex": data[i]['label'],
            "predLabel": labels[predIndex],
            "originalLabel": labels[data[i]['label']],
            "inftime": inftime,
            "preprocess": preproctime
        }

        // console.log(resp_obj)

        predData.push(resp_obj)
    }
    var total_inf = now() - total_start
    var avg_inf_speed = total_inf / data.length
    var inf_ps = 1000 / avg_inf_speed

    var acc = utils.calculateAccuracy(predData)

    response = {
        "loadtime": loadtime.reduce((a, b) => a + b, 0),
        "read_wasm": loadtime[0],
        "load_weights": loadtime[1],
        "pop_weights": loadtime[2],
        "predictions": predData,
        "accuracy_top1": acc,
        "avg_inf_speed": avg_inf_speed,
        "inference_ps": inf_ps
    }

    res.send(JSON.stringify(response));

    // debug.writeToFile(
    //     each, ['label', 'loadtime', 'inftime', 'read_wasm', 'load_weights', 'pop_weights', "preprocess"],
    //     'results/' + modelType + "/" + variantType + '.csv'
    // )
});

module.exports = router;