var preProcLib = require('../utils/preprocessor.js')

module.exports = {
    collectModel: function(modelType, variantType) {

        //This function owns a database that links a modelType string passed User's post call to the function that contains the model's data

        modelDatabase = {
            "mobilenetv2": MobileNetv2,
            "tiny": TinyModel,
        }
        return modelDatabase[modelType](variantType)

    }
}

function MobileNetv2(variantType) {
    // The base path is the path to the folder containing the params and graph files
    // The modelObj provides filenames relative to the basepath collected from ModelInfo
    const modelInfo = {
        "baseline": { "base": "public/wasm/mobilenetv2/baseline/", "input_type": "float32", "preprocessor": preProcLib.preprocess_imagenet },
        "reimann": { "base": "public/wasm/mobilenetv2/reimann/", "input_type": "float32", "preprocessor": preProcLib.preprocess_imagenet },
        "lorenz_int8": { "base": "public/wasm/mobilenetv2/lorenz_int8/", "input_type": "uint8", "preprocessor": preProcLib.preprocess_uint8 },
        "lorenz_int16": { "base": "public/wasm/mobilenetv2/lorenz_int16/", "input_type": "uint8", "preprocessor": preProcLib.preprocess_uint8 }
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
        "baseline": { "base": "public/wasm/tiny/baseline/", "input_type": "float32", "preprocessor": preProcLib.preprocess_imagenet },
        "reimann": { "base": "public/wasm/tiny/reimann/", "input_type": "float32", "preprocessor": preProcLib.preprocess_imagenet },
        "lorenz_int8": { "base": "public/wasm/tiny/lorenz_int8/", "input_type": "uint8", "preprocessor": preProcLib.preprocess_uint8 },
        "lorenz_int16": { "base": "public/wasm/tiny/lorenz_int16/", "input_type": "uint8", "preprocessor": preProcLib.preprocess_uint8 }
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