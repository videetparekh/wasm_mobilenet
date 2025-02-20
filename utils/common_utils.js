const fs = require("fs");

module.exports = {
    getLabels: async function(labelsFile) {
        const synset = await JSON.parse(fs.readFileSync(labelsFile));
        return synset
    },

    calculateAccuracy: function(predictionData) {
        var counter = 0
        for (var i = 0; i < predictionData.length; i++) {
            if (predictionData[i]['predIndex'] == predictionData[i]['originalIndex']) {
                counter++;
            }
        }
        console.log(counter.toString() + "/" + predictionData.length.toString())
        return counter / predictionData.length;
    }
}