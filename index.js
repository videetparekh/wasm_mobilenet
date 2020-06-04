/**
 * Required External Modules
 */
const express = require("express");
const path = require("path");
var indexRouter = require('./routes/index');
var predictRouter = require('./routes/predict')
var evaluateRouter = require('./routes/evaluate')

/**
 * App Variables
 */
const app = express();
const PORT = process.env.PORT || "8000";
const HOST = '0.0.0.0';

/**
 *  App Configuration
 */

app.set("views", path.join(__dirname, "views"));
app.set("view engine", "pug");
app.use(express.static(path.join(__dirname, "public")));
app.use(express.json({ limit: "100mb", extended: true }));
app.use(express.urlencoded({ limit: "100mb", extended: true }));

/**
 * Routes Definitions
 */
app.use("/", indexRouter);
app.use("/predict", predictRouter);
app.use("/evaluate", evaluateRouter);
/**
 * Server Activation
 */
app.listen(PORT, HOST, () => {
    console.log(`Listening to requests on http://${HOST}:${PORT}`);
});