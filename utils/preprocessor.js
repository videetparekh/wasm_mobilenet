const Canvas = require("canvas")

module.exports = {
    preprocess_imagenet: function(imageData) {
        var rgbFP32 = new Float32Array(cleanAndStripAlpha(imageData))
        var normArray = rgbFP32.map(normalize)
        return rgbFP32
    },

    preprocess_uint8: function(imageData) {
        var rgbU8 = cleanAndStripAlpha(imageData)
        return rgbU8;
    },

    drawCanvas: function(url, shape) {
        var cvs = Canvas.createCanvas(shape[2], shape[3]);
        var context = cvs.getContext('2d');
        var img = new Canvas.Image;
        img.onload = function() {
            context.drawImage(img, 0, 0);
        };
        img.src = url;
        return context.getImageData(0, 0, shape[2], shape[3])
    }
}

/*
Expectation for a Preprocessor function:

Function expects (imageData: output of drawCanvas function) { returns Array} 
*/

function cleanAndStripAlpha(imageData) {
    const width = imageData.width;
    const height = imageData.height;
    const npixels = width * height;

    const rgbaU8 = imageData.data;

    // Drop alpha channel
    const rgbU8 = new Uint8Array(npixels * 3);
    for (let i = 0; i < npixels; ++i) {
        rgbU8[i * 3] = rgbaU8[i * 4];
        rgbU8[i * 3 + 1] = rgbaU8[i * 4 + 1];
        rgbU8[i * 3 + 2] = rgbaU8[i * 4 + 2];
    }
    return rgbU8;
}

function normalize(a) {
    return ((a / 127.5) - 1.0)
}