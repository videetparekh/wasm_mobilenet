var stringify = require('csv-stringify');
const fs = require("fs");

module.exports = {
    writeToFile: function(data, columnHeaders, fileName) {
        stringify(data, { header: true, columns: columnHeaders }, function(err, output) {
            fs.writeFile(fileName, output, 'utf8', function(err) {
                if (err) {
                    console.log('Some error occured - file either not saved or corrupted file saved.');
                } else {
                    console.log('It\'s saved!');
                }
            });
        });
    }
}