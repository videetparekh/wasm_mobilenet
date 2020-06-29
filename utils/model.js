var preproc = require('../utils/preprocessor.js')

module.exports = {
    collectModel: function(modelType, variantType) {

        //This function owns a database that links a modelType string passed User's post call to the function that contains the model's data
        const modelDatabase = getModelDatabase();
        return modelDatabase[modelType](variantType)

    },

    getModelVariantList: function() {
        var modelKeys = Object.keys(getModelDatabase())
        var variantKeys = ['fp32', 'int8', 'int16']
        return { "modelTypes": modelKeys, "variantTypes": variantKeys }
    }
}

// Link Model names to their functions here
function getModelDatabase() {
    modelDatabase = {
        "mobilenetv2": MobileNetv2,
        "tiny": TinyModel
    }
    return modelDatabase;
}

function MobileNetv2(variantType) {
    // The base path is the path to the folder containing the params and graph files
    // The modelObj provides filenames relative to the basepath collected from ModelInfo
    const modelInfo = {
        "fp32": { "base": "public/wasm/mobilenetv2/fp32/", "input_type": "float32", "preprocessor": preproc.preprocess_imagenet },
        "int8": { "base": "public/wasm/mobilenetv2/int8/", "input_type": "uint8", "preprocessor": preproc.preprocess_uint8 },
        "int16": { "base": "public/wasm/mobilenetv2/int16/", "input_type": "uint8", "preprocessor": preproc.preprocess_uint8 }
    }

    var modelObj = {
        "graph": "modelDescription.json",
        "params": "modelParams.params",
        "wasm": "modelLibrary.wasm",
        "labels": "public/wasm/open_images_labels.json",
        "input_shape": [1, 3, 224, 224],
        "input_name": "input_1",
        "output_shape": [1, 10]
    }

    modelObj = Object.assign(modelObj, modelInfo[variantType])

    return modelObj
}

function TinyModel(variantType) {
    // The base path is the path to the folder containing the params and graph files
    // The modelObj provides filenames relative to the basepath collected from ModelInfo
    const modelInfo = {
        "fp32": { "base": "public/wasm/tiny/fp32/", "input_type": "float32", "preprocessor": preproc.preprocess_imagenet },
        "int8": { "base": "public/wasm/tiny/int8/", "input_type": "uint8", "preprocessor": preproc.preprocess_uint8 },
        "int16": { "base": "public/wasm/tiny/int16/", "input_type": "uint8", "preprocessor": preproc.preprocess_uint8 }
    }

    var modelObj = {
        "graph": "modelDescription.json",
        "params": "modelParams.params",
        "wasm": "modelLibrary.wasm",
        "labels": "public/wasm/person_classifier_labels.json",
        "input_shape": [1, 3, 96, 96],
        "input_name": "MobilenetV2/input",
        "output_shape": [1, 2]
    }

    modelObj = Object.assign(modelObj, modelInfo[variantType])

    return modelObj
}

/*
Expectation for a Model Schema:

Function expects (VariantType: string) { returns modelObj} 
*/