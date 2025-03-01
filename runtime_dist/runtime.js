"use strict";
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.instantiate = exports.Instance = exports.Module = exports.NDArray = exports.DLDataType = exports.DLContext = exports.Scalar = void 0;
const memory_1 = require("./memory");
const support_1 = require("./support");
const environment_1 = require("./environment");
const webgpu_1 = require("./webgpu");
const compact = require("./compact");
/**
 * @internal
 * FFI Library wrapper, maintains most runtime states.
 */
class FFILibrary {
    constructor(wasmInstance, imports) {
        this.recycledCallStacks = [];
        this.wasmInstance = wasmInstance;
        this.memory = new memory_1.Memory(this.detectWasmMemory(this.wasmInstance, imports));
        support_1.assert(this.wasmInstance.exports !== undefined, "Expect the library module contains exports");
        this.exports = this.wasmInstance.exports;
        this.wasm32 = this.memory.wasm32;
        this.validateInstance();
    }
    dispose() {
        while (this.recycledCallStacks.length != 0) {
            this.recycledCallStacks.pop().dispose();
        }
    }
    sizeofPtr() {
        return this.memory.sizeofPtr();
    }
    checkCall(code) {
        if (code != 0) {
            const msgPtr = this.exports
                .TVMGetLastError();
            throw new Error("TVMError: " + this.memory.loadCString(msgPtr));
        }
    }
    getOrAllocCallStack() {
        if (this.recycledCallStacks.length != 0) {
            return this.recycledCallStacks.pop();
        }
        return new memory_1.CachedCallStack(this.memory, this.exports.TVMWasmAllocSpace, this.exports.TVMWasmFreeSpace);
    }
    recycleCallStack(callstack) {
        callstack.reset();
        this.recycledCallStacks.push(callstack);
    }
    validateInstance() {
        this.checkExports(["TVMWasmAllocSpace", "TVMWasmFreeSpace", "TVMFuncFree"]);
    }
    checkExports(funcNames) {
        const missList = [];
        for (const name of funcNames) {
            const f = this.exports[name];
            if (!(f instanceof Function)) {
                missList.push(name);
            }
        }
        if (missList.length != 0) {
            throw new Error("Cannot find " + missList + " in exports");
        }
    }
    detectWasmMemory(instance, imports) {
        if (instance.exports.memory instanceof WebAssembly.Memory) {
            return instance.exports.memory;
        }
        if (imports.env && imports.env.memory instanceof WebAssembly.Memory) {
            return imports.env.memory;
        }
        throw new Error("Cannt detect wasm memory from imports " +
            imports +
            " or exports" +
            instance.exports);
    }
}
/**
 * A typed scalar constant used to represent a typed number
 * argument to PackedFunc calls.
 */
class Scalar {
    constructor(value, dtype) {
        this.value = value;
        this.dtype = dtype;
    }
}
exports.Scalar = Scalar;
/**
 * Cell holds the PackedFunc object.
 */
class PackedFuncCell {
    constructor(handle, lib) {
        this.handle = handle;
        this.lib = lib;
    }
    dispose() {
        if (this.handle != 0) {
            this.lib.checkCall(this.lib.exports.TVMFuncFree(this.handle));
            this.handle = 0;
        }
    }
}
const DeviceEnumToStr = {
    1: "cpu",
    2: "gpu",
    4: "opencl",
    8: "metal",
    15: "webgpu"
};
const DeviceStrToEnum = {
    cpu: 1,
    gpu: 2,
    cuda: 2,
    cl: 4,
    opencl: 4,
    vulkan: 7,
    metal: 8,
    webgpu: 15
};
/**
 * Represent a runtime context where a NDArray can reside.
 */
class DLContext {
    constructor(deviceType, deviceId, lib) {
        const tp = typeof deviceType;
        if (tp == "string") {
            this.deviceType = DeviceStrToEnum[deviceType];
            if (this.deviceType == undefined) {
                throw new Error("Cannot recogonize deviceType " + deviceType);
            }
        }
        else if (tp == "number") {
            this.deviceType = deviceType;
        }
        else {
            throw new Error("Cannot take type " + tp + " as deviceType");
        }
        this.deviceId = deviceId;
        this.lib = lib;
    }
    /**
     * Synchronize the context
     */
    sync() {
        return __awaiter(this, void 0, void 0, function* () {
            if (this.deviceType == DeviceStrToEnum.webgpu) {
                support_1.assert(this.lib.webGPUContext !== undefined);
                yield this.lib.webGPUContext.sync();
            }
        });
    }
    toString() {
        return (DeviceEnumToStr[this.deviceType] + "(" + this.deviceId.toString() + ")");
    }
}
exports.DLContext = DLContext;
const DLDataTypeCodeToStr = {
    0: "int",
    1: "uint",
    2: "float",
    4: "handle",
};
/**
 * Runtime data type of NDArray.
 */
class DLDataType {
    constructor(code, bits, lanes) {
        this.code = code;
        this.bits = bits;
        this.lanes = lanes;
    }
    toString() {
        const ret = DLDataTypeCodeToStr[this.code] + this.bits.toString();
        if (this.lanes != 1) {
            return ret + "x" + this.lanes.toString();
        }
        else {
            return ret;
        }
    }
    numStorageBytes() {
        return (this.bits * this.lanes + 7) >> 3;
    }
}
exports.DLDataType = DLDataType;
/**
 * n-dimnesional array.
 */
class NDArray {
    constructor(handle, isView, lib) {
        this.handle = handle;
        this.isView = isView;
        this.lib = lib;
        if (this.isView) {
            this.dltensor = handle;
        }
        else {
            this.dltensor = this.getDLTensorFromArrayHandle(this.handle);
        }
        // constant offsets.
        const arrayOffsetData = 0;
        const arrayOffsetContext = arrayOffsetData + this.lib.sizeofPtr();
        const arrayOffsetDevType = arrayOffsetContext;
        const arrayOffsetDevId = arrayOffsetContext + 4 /* I32 */;
        const arrayOffsetNdim = arrayOffsetContext + 8 /* DLContext */;
        const arrayOffsetDtype = arrayOffsetNdim + 4 /* I32 */;
        const arrayOffsetDtypeCode = arrayOffsetDtype;
        const arrayOffsetDtypeBits = arrayOffsetDtype + 1 /* U8 */;
        const arrayOffsetDtypeLanes = arrayOffsetDtypeBits + 1 /* U8 */;
        const arrayOffsetShape = arrayOffsetDtype + 4 /* DLDataType */;
        const arrayOffsetStrides = arrayOffsetShape + this.lib.sizeofPtr();
        const arrayOffsetByteOffset = arrayOffsetStrides + this.lib.sizeofPtr();
        // dataPtr
        this.dataPtr = lib.memory.loadPointer(this.dltensor);
        // ndim
        this.ndim = lib.memory.loadI32(this.dltensor + arrayOffsetNdim);
        // shape
        const cshapePtr = lib.memory.loadPointer(this.dltensor + arrayOffsetShape);
        this.shape = [];
        for (let i = 0; i < this.ndim; ++i) {
            this.shape.push(lib.memory.loadI64(cshapePtr + i * 8 /* I64 */));
        }
        // dtype
        const code = lib.memory.loadU8(this.dltensor + arrayOffsetDtypeCode);
        const bits = lib.memory.loadU8(this.dltensor + arrayOffsetDtypeBits);
        const lanes = lib.memory.loadU16(this.dltensor + arrayOffsetDtypeLanes);
        this.dlDataType = new DLDataType(code, bits, lanes);
        this.dtype = this.dlDataType.toString();
        // ctx
        const deviceType = lib.memory.loadI32(this.dltensor + arrayOffsetDevType);
        const deviceId = lib.memory.loadI32(this.dltensor + arrayOffsetDevId);
        this.context = new DLContext(deviceType, deviceId, lib);
        // byte_offset
        this.byteOffset = lib.memory.loadI64(this.dltensor + arrayOffsetByteOffset);
    }
    dispose() {
        if (this.handle != 0 && !this.isView) {
            this.lib.checkCall(this.lib.exports.TVMArrayFree(this.handle));
            this.handle = 0;
        }
    }
    /**
     * Copy data from another NDArray or javascript array.
     * The number of elements must match.
     *
     * @param data The source data array.
     * @returns this
     */
    copyFrom(data) {
        if (data instanceof NDArray) {
            this.lib.checkCall(this.lib.exports.TVMArrayCopyFromTo(data.handle, this.handle, 0));
            return this;
        }
        else {
            const size = this.shape.reduce((a, b) => {
                return a * b;
            }, 1);
            if (data.length != size) {
                throw new Error("data size and shape mismatch data.length" +
                    data.length +
                    " vs " +
                    size);
            }
            let buffer;
            if (this.dtype == "float32") {
                buffer = Float32Array.from(data).buffer;
            }
            else if (this.dtype == "float64") {
                buffer = Float64Array.from(data).buffer;
            }
            else if (this.dtype == "int32") {
                buffer = Int32Array.from(data).buffer;
            }
            else if (this.dtype == "int8") {
                buffer = Int8Array.from(data).buffer;
            }
            else if (this.dtype == "uint8") {
                buffer = Uint8Array.from(data).buffer;
            }
            else {
                throw new Error("Unsupported data type " + this.dtype);
            }
            return this.copyFromRawBytes(new Uint8Array(buffer));
        }
    }
    /**
     * Copy data from raw bytes.
     * @param data Uint8Array of bytes.
     * @returns this
     */
    copyFromRawBytes(data) {
        const size = this.shape.reduce((a, b) => {
            return a * b;
        }, 1);
        const nbytes = this.dlDataType.numStorageBytes() * size;
        if (nbytes != data.length) {
            throw new Error("Expect the data's length equals nbytes=" + nbytes);
        }
        const stack = this.lib.getOrAllocCallStack();
        const tempOffset = stack.allocRawBytes(nbytes);
        const tempPtr = stack.ptrFromOffset(tempOffset);
        this.lib.memory.storeRawBytes(tempPtr, data);
        this.lib.checkCall(this.lib.exports.TVMArrayCopyFromBytes(this.handle, tempPtr, nbytes));
        this.lib.recycleCallStack(stack);
        return this;
    }
    /**
     * Return a copied Uint8Array of the raw bytes in the NDArray.
     * @returns The result array.
     */
    toRawBytes() {
        if (this.context.deviceType != DeviceStrToEnum.cpu) {
            throw new Error("Can only synchronize copy for GPU array, use copyfrom instead.");
        }
        const size = this.shape.reduce((a, b) => {
            return a * b;
        }, 1);
        const nbytes = this.dlDataType.numStorageBytes() * size;
        const stack = this.lib.getOrAllocCallStack();
        const tempOffset = stack.allocRawBytes(nbytes);
        const tempPtr = stack.ptrFromOffset(tempOffset);
        this.lib.checkCall(this.lib.exports.TVMArrayCopyToBytes(this.handle, tempPtr, nbytes));
        const ret = this.lib.memory.loadRawBytes(tempPtr, nbytes);
        this.lib.recycleCallStack(stack);
        return ret;
    }
    /**
     * Return a TypedArray copy of the NDArray, the specific type depends on
     * the dtype of the NDArray.
     * @returns The result array.
     */
    toArray() {
        const stype = this.dtype;
        if (stype == "float32") {
            return new Float32Array(this.toRawBytes().buffer);
        }
        else if (stype == "float64") {
            return new Float64Array(this.toRawBytes().buffer);
        }
        else if (stype == "int32") {
            return new Int32Array(this.toRawBytes().buffer);
        }
        else if (stype == "int8") {
            return new Int8Array(this.toRawBytes().buffer);
        }
        else if (stype == "uint8") {
            return new Uint8Array(this.toRawBytes().buffer);
        }
        else {
            throw new Error("Unsupported data type " + this.dtype);
        }
    }
    getDLTensorFromArrayHandle(handle) {
        // Note: this depends on the NDArray C ABI.
        // keep this function in case of ABI change.
        return handle;
    }
}
exports.NDArray = NDArray;
/**
 * Runtime Module.
 */
class Module {
    constructor(handle, lib, makePackedFunc) {
        this.handle = handle;
        this.lib = lib;
        this.makePackedFunc = makePackedFunc;
    }
    dispose() {
        if (this.handle != 0) {
            this.lib.checkCall(this.lib.exports.TVMModFree(this.handle));
            this.handle = 0;
        }
    }
    /**
     * Get a function in the module.
     * @param name The name of the function.
     * @returns The result function.
     */
    getFunction(name) {
        const stack = this.lib.getOrAllocCallStack();
        const nameOffset = stack.allocRawBytes(name.length + 1);
        stack.storeRawBytes(nameOffset, support_1.StringToUint8Array(name));
        const outOffset = stack.allocPtrArray(1);
        const outPtr = stack.ptrFromOffset(outOffset);
        stack.commitToWasmMemory(outOffset);
        this.lib.checkCall(this.lib.exports.TVMModGetFunction(this.handle, stack.ptrFromOffset(nameOffset), 1, outPtr));
        const handle = this.lib.memory.loadPointer(outPtr);
        this.lib.recycleCallStack(stack);
        if (handle == 0) {
            throw Error("Cannot find function " + name);
        }
        const ret = this.makePackedFunc(handle);
        return ret;
    }
    /**
     * Import another module into the current runtime module.
     * @param mod The module to be imported.
     */
    importModule(mod) {
        this.lib.checkCall(this.lib.exports.TVMModImport(this.handle, mod.handle));
    }
}
exports.Module = Module;
/**
 *  Graph runtime.
 *
 *  This is a thin wrapper of the underlying TVM module.
 *  you can also directly call set_input, run, and get_output
 *  of underlying module functions
 */
class GraphRuntime {
    /**
     * COnstructor
     * @param module The underlying module.
     */
    constructor(module) {
        this.module = module;
        this.packedSetInput = module.getFunction("set_input");
        this.packedRun = module.getFunction("run");
        this.packedGetOutput = module.getFunction("get_output");
        this.packedLoadParams = module.getFunction("load_params");
    }
    dispose() {
        this.packedSetInput.dispose();
        this.packedRun.dispose();
        this.packedGetOutput.dispose();
    }
    /**
     * Set input to the executor.
     *
     * @param key The input key.
     * @param value The value to get set.
     */
    setInput(key, value) {
        if (typeof key == "number") {
            this.packedSetInput(new Scalar(key, "int32"), value);
        }
        else {
            this.packedSetInput(key, value);
        }
    }
    /**
     * Execute the underlying graph.
     */
    run() {
        this.packedRun();
    }
    /**
     * Get index-th output.
     * @param index The index number.
     * @param out The optional output storage parameters.
     * @returns The output array.
     */
    getOutput(index, out = undefined) {
        if (out !== undefined) {
            this.packedGetOutput(new Scalar(index, "int32"), out);
            return out;
        }
        else {
            return this.packedGetOutput(new Scalar(index, "int32"));
        }
    }
    /**
     * Load parameters from parameter binary.
     * @param paramBinary The parameter binary.
     */
    loadParams(paramBinary) {
        this.packedLoadParams(paramBinary);
    }
    /**
     * Benchmark stable execution of the graph(without data copy).
     * @params ctx The context to sync during each run.
     * @number The number of times to compute the average.
     * @repeat The number of times to repeat the run.
     */
    benchmarkRuns(ctx, number = 10, repeat = 4) {
        return __awaiter(this, void 0, void 0, function* () {
            // Skip first run as it can involve GPU warmup and module loading time.
            const perf = compact.getPeformance();
            const results = [];
            this.run();
            yield ctx.sync();
            for (let k = 0; k < repeat; ++k) {
                const tstart = perf.now();
                for (let i = 0; i < number; ++i) {
                    this.run();
                }
                yield ctx.sync();
                const tend = perf.now();
                results.push((tend - tstart) / number);
            }
            return results;
        });
    }
}
/**
 * TVM runtime instance.
 */
class Instance {
    /**
     * Constructor
     *
     * importObject can also be a {@link LibraryProvider} object,
     * a WASI object, or an object containing wasmLibraryProvider field.
     *
     * @param wasmModule The input module or instance.
     * @param importObject The imports to initialize the wasmInstance if it is not provided.
     * @param wasmInstance Additional wasm instance argument for deferred construction.
     * @param env Directly specified environment module.
     *
     * @see Please use the async version {@link instantiate} when targeting browsers.
     */
    constructor(wasmModule, importObject = {}, wasmInstance, env) {
        if (wasmInstance instanceof WebAssembly.Instance) {
            support_1.assert(env instanceof environment_1.Environment, "env must be provided when passing in instance");
        }
        else {
            support_1.assert(env === undefined);
            env = new environment_1.Environment(importObject);
            wasmInstance = new WebAssembly.Instance(wasmModule, env.imports);
        }
        env.start(wasmInstance);
        this.env = env;
        this.lib = new FFILibrary(wasmInstance, env.imports);
        this.memory = this.lib.memory;
        this.exports = this.lib.exports;
        this.registerEnvGlobalPackedFuncs();
    }
    dispose() {
        this.lib.dispose();
    }
    /**
     * Get system-wide library module in the wasm.
     * System lib is a global module that contains self register functions in startup.
     * @returns The system library module.
     */
    systemLib() {
        const getSysLib = this.getGlobalFunc("runtime.SystemLib");
        const mod = getSysLib();
        getSysLib.dispose();
        return mod;
    }
    /**
     * List all the global function names registered in the runtime.
     * @returns The name list.
     */
    listGlobalFuncNames() {
        const stack = this.lib.getOrAllocCallStack();
        const outSizeOffset = stack.allocPtrArray(2);
        const outSizePtr = stack.ptrFromOffset(outSizeOffset);
        const outArrayPtr = stack.ptrFromOffset(outSizeOffset + this.lib.sizeofPtr());
        this.lib.checkCall(this.exports.TVMFuncListGlobalNames(outSizePtr, outArrayPtr));
        const size = this.memory.loadI32(outSizePtr);
        const array = this.memory.loadPointer(outArrayPtr);
        const names = [];
        for (let i = 0; i < size; ++i) {
            names.push(this.memory.loadCString(this.memory.loadPointer(array + this.lib.sizeofPtr() * i)));
        }
        this.lib.recycleCallStack(stack);
        return names;
    }
    /**
     * Register function to be global function in tvm runtime.
     * @param name The name of the function.
     * @param f function to be registered.
     * @param override Whether overwrite function in existing registry.
     */
    registerFunc(name, func, override = false) {
        const packedFunc = this.toPackedFunc(func);
        const ioverride = override ? 1 : 0;
        const stack = this.lib.getOrAllocCallStack();
        const nameOffset = stack.allocRawBytes(name.length + 1);
        stack.storeRawBytes(nameOffset, support_1.StringToUint8Array(name));
        stack.commitToWasmMemory();
        this.lib.checkCall(this.lib.exports.TVMFuncRegisterGlobal(stack.ptrFromOffset(nameOffset), packedFunc._tvmPackedCell.handle, ioverride));
    }
    /**
     * Get global PackedFunc from the runtime.
     * @param name The name of the function.
     * @returns The result function.
     */
    getGlobalFunc(name) {
        const stack = this.lib.getOrAllocCallStack();
        const nameOffset = stack.allocRawBytes(name.length + 1);
        stack.storeRawBytes(nameOffset, support_1.StringToUint8Array(name));
        const outOffset = stack.allocPtrArray(1);
        const outPtr = stack.ptrFromOffset(outOffset);
        stack.commitToWasmMemory(outOffset);
        this.lib.checkCall(this.exports.TVMFuncGetGlobal(stack.ptrFromOffset(nameOffset), outPtr));
        const handle = this.memory.loadPointer(outPtr);
        this.lib.recycleCallStack(stack);
        if (handle == 0) {
            throw Error("Cannot find global function " + name);
        }
        const ret = this.makePackedFunc(handle);
        return ret;
    }
    /**
     * Check if func is PackedFunc.
     *
     * @param func The input.
     * @returns The check result.
     */
    isPackedFunc(func) {
        // eslint-disable-next-line no-prototype-builtins
        return typeof func == "function" && func.hasOwnProperty("_tvmPackedCell");
    }
    /**
     * Convert func to PackedFunc
     *
     * @param func Input function.
     * @returns The converted function.
     */
    toPackedFunc(func) {
        if (this.isPackedFunc(func))
            return func;
        return this.createPackedFuncFromCFunc(this.wrapJSFuncAsPackedCFunc(func));
    }
    /**
     * Convert dtype to {@link DLDataType}
     *
     * @param dtype The input dtype string or DLDataType.
     * @returns The converted result.
     */
    toDLDataType(dtype) {
        if (dtype instanceof DLDataType)
            return dtype;
        if (typeof dtype == "string") {
            let pattern = dtype;
            let code, bits = 32, lanes = 1;
            if (pattern.substring(0, 5) == "float") {
                pattern = pattern.substring(5, pattern.length);
                code = 2 /* Float */;
            }
            else if (pattern.substring(0, 3) == "int") {
                pattern = pattern.substring(3, pattern.length);
                code = 0 /* Int */;
            }
            else if (pattern.substring(0, 4) == "uint") {
                pattern = pattern.substring(4, pattern.length);
                code = 1 /* UInt */;
            }
            else if (pattern.substring(0, 6) == "handle") {
                pattern = pattern.substring(5, pattern.length);
                code = 3 /* TVMOpaqueHandle */;
                bits = 64;
            }
            else {
                throw new Error("Unknown dtype " + dtype);
            }
            const arr = pattern.split("x");
            if (arr.length >= 1) {
                const parsed = parseInt(arr[0]);
                if (parsed + "" == arr[0]) {
                    bits = parsed;
                }
            }
            if (arr.length >= 2) {
                lanes = parseInt(arr[1]);
            }
            return new DLDataType(code, bits, lanes);
        }
        else {
            throw new Error("Unknown dtype " + dtype);
        }
    }
    /**
     * Create a new {@link Scalar} that can be passed to a PackedFunc.
     * @param value The number value.
     * @param dtype The dtype string.
     * @returns The created scalar.
     */
    scalar(value, dtype) {
        return new Scalar(value, dtype);
    }
    /**
     * Create a new {@link DLContext}
     * @param deviceType The device type.
     * @param deviceId The device index.
     * @returns The created context.
     */
    context(deviceType, deviceId = 0) {
        return new DLContext(deviceType, deviceId, this.lib);
    }
    /**
     * Create a new cpu {@link DLContext}
     * @param deviceId The device index.
     */
    cpu(deviceId = 0) {
        return this.context("cpu", deviceId);
    }
    /**
     * Create a new webgpu {@link DLContext}
     * @param deviceId The device index.
     */
    webgpu(deviceId = 0) {
        return this.context("webgpu", deviceId);
    }
    /**
     * Create an empty {@link NDArray} with given shape and dtype.
     *
     * @param shape The shape of the array.
     * @param dtype The data type of the array.
     * @param ctx The context of the ndarray.
     * @returns The created ndarray.
     */
    empty(shape, dtype = "float32", ctx = this.context("cpu", 0)) {
        dtype = this.toDLDataType(dtype);
        shape = typeof shape == "number" ? [shape] : shape;
        const stack = this.lib.getOrAllocCallStack();
        const shapeOffset = stack.allocRawBytes(shape.length * 8 /* I64 */);
        for (let i = 0; i < shape.length; ++i) {
            stack.storeI64(shapeOffset + i * 8 /* I64 */, shape[i]);
        }
        const outOffset = stack.allocPtrArray(1);
        const outPtr = stack.ptrFromOffset(outOffset);
        stack.commitToWasmMemory(outOffset);
        this.lib.checkCall(this.exports.TVMArrayAlloc(stack.ptrFromOffset(shapeOffset), shape.length, dtype.code, dtype.bits, dtype.lanes, ctx.deviceType, ctx.deviceId, outPtr));
        const ret = new NDArray(this.memory.loadPointer(outPtr), false, this.lib);
        this.lib.recycleCallStack(stack);
        return ret;
    }
    /**
     * Create a new graph runtime.
     *
     * @param graphJson The graph runtime json file.
     * @param lib The underlying library.
     * @param ctx The execution context of the graph.
     */
    createGraphRuntime(graphJson, lib, ctx) {
        const fcreate = this.getGlobalFunc("tvm.graph_runtime.create");
        const module = fcreate(graphJson, lib, this.scalar(ctx.deviceType, "int32"), this.scalar(ctx.deviceId, "int32"));
        return new GraphRuntime(module);
    }
    /**
     * Register an asyncfunction to be global function in the server.
     * @param name The name of the function.
     * @param func function to be registered.
     * @param override Whether overwrite function in existing registry.
     *
     * @note The async function will only be used for serving remote calls in the rpc.
     */
    registerAsyncServerFunc(name, func, override = false) {
        const asyncVariant = (...args) => {
            const fargs = args.slice(0, args.length - 1);
            const callback = args[args.length - 1];
            const promise = func(...fargs);
            promise.then((rv) => {
                callback(this.scalar(4 /* kReturn */, "int32"), rv);
            });
        };
        this.registerFunc("__async." + name, asyncVariant, override);
    }
    /**
     * Initialize webgpu in the runtime.
     * @param device The given GPU device.
     */
    initWebGPU(device) {
        const webGPUContext = new webgpu_1.WebGPUContext(this.memory, device);
        this.registerFunc("wasm.WebGPUDeviceAPI", (name) => {
            return webGPUContext.getDeviceAPI(name);
        });
        this.registerFunc("wasm.WebGPUCreateShader", (info, data) => {
            return webGPUContext.createShader(info, data);
        });
        this.registerAsyncServerFunc("wasm.WebGPUWaitForTasks", () => __awaiter(this, void 0, void 0, function* () {
            yield webGPUContext.sync();
        }));
        this.lib.webGPUContext = webGPUContext;
    }
    /** Register global packed functions needed by the backend to the env. */
    registerEnvGlobalPackedFuncs() {
        // Register the timer function to enable the time_evaluator.
        const perf = compact.getPeformance();
        // Helper function to time the finvoke
        const timeExecution = (finvoke, ctx, nstep, repeat, minRepeatMs) => __awaiter(this, void 0, void 0, function* () {
            finvoke(this.scalar(1, "int32"));
            yield ctx.sync();
            const result = [];
            let setupNumber = nstep;
            for (let i = 0; i < repeat; ++i) {
                let durationMs = 0.0;
                do {
                    if (durationMs > 0.0) {
                        setupNumber = Math.floor(Math.max(minRepeatMs / (durationMs / nstep) + 1, nstep * 1.618));
                    }
                    const tstart = perf.now();
                    finvoke(this.scalar(setupNumber, "int32"));
                    yield ctx.sync();
                    const tend = perf.now();
                    durationMs = tend - tstart;
                } while (durationMs < minRepeatMs);
                const speed = durationMs / setupNumber / 1000;
                result.push(speed);
            }
            const ret = new Float64Array(result.length);
            ret.set(result);
            return new Uint8Array(ret.buffer);
        });
        const addOne = (x) => __awaiter(this, void 0, void 0, function* () {
            yield new Promise(resolve => setTimeout(resolve, 100));
            return x + 1;
        });
        this.registerAsyncServerFunc("wasm.TimeExecution", timeExecution);
        this.registerAsyncServerFunc("testing.asyncAddOne", addOne);
    }
    createPackedFuncFromCFunc(func) {
        let findex = this.env.packedCFuncTable.length;
        if (this.env.packedCFuncTableFreeId.length != 0) {
            findex = this.env.packedCFuncTableFreeId.pop();
        }
        else {
            this.env.packedCFuncTable.push(undefined);
        }
        this.env.packedCFuncTable[findex] = func;
        const stack = this.lib.getOrAllocCallStack();
        const outOffset = stack.allocPtrArray(1);
        const outPtr = stack.ptrFromOffset(outOffset);
        this.lib.checkCall(this.exports
            .TVMWasmFuncCreateFromCFunc(findex, outPtr));
        const ret = this.makePackedFunc(this.memory.loadPointer(outPtr));
        this.lib.recycleCallStack(stack);
        return ret;
    }
    /**
     * Set packed function arguments into the location indicated by argsValue and argsCode.
     * Allocate new temporary space from the stack if necessary.
     *
     * @parma stack The call stack
     * @param args  The input arguments.
     * @param argsValue The offset of argsValue.
     * @param argsCode The offset of argsCode.
     */
    setPackedArguments(stack, args, argsValue, argsCode) {
        for (let i = 0; i < args.length; ++i) {
            let val = args[i];
            const tp = typeof val;
            const valueOffset = argsValue + i * 8 /* TVMValue */;
            const codeOffset = argsCode + i * 4 /* I32 */;
            if (val instanceof NDArray) {
                stack.storePtr(valueOffset, val.handle);
                stack.storeI32(codeOffset, 13 /* TVMNDArrayHandle */);
            }
            else if (val instanceof Scalar) {
                if (val.dtype.startsWith("int") || val.dtype.startsWith("uint")) {
                    stack.storeI64(valueOffset, val.value);
                    stack.storeI32(codeOffset, 0 /* Int */);
                }
                else if (val.dtype.startsWith("float")) {
                    stack.storeF64(valueOffset, val.value);
                    stack.storeI32(codeOffset, 2 /* Float */);
                }
                else {
                    support_1.assert(val.dtype == "handle", "Expect handle");
                    stack.storePtr(valueOffset, val.value);
                    stack.storeI32(codeOffset, 3 /* TVMOpaqueHandle */);
                }
            }
            else if (val instanceof DLContext) {
                stack.storeI32(valueOffset, val.deviceType);
                stack.storeI32(valueOffset + 4 /* I32 */, val.deviceType);
                stack.storeI32(codeOffset, 6 /* TVMContext */);
            }
            else if (tp == "number") {
                stack.storeF64(valueOffset, val);
                stack.storeI32(codeOffset, 2 /* Float */);
                // eslint-disable-next-line no-prototype-builtins
            }
            else if (tp == "function" && val.hasOwnProperty("_tvmPackedCell")) {
                stack.storePtr(valueOffset, val._tvmPackedCell.handle);
                stack.storeI32(codeOffset, 10 /* TVMPackedFuncHandle */);
            }
            else if (val === null || val == undefined) {
                stack.storePtr(valueOffset, 0);
                stack.storeI32(codeOffset, 4 /* Null */);
            }
            else if (tp == "string") {
                stack.allocThenSetArgString(valueOffset, val);
                stack.storeI32(codeOffset, 11 /* TVMStr */);
            }
            else if (val instanceof Uint8Array) {
                stack.allocThenSetArgBytes(valueOffset, val);
                stack.storeI32(codeOffset, 12 /* TVMBytes */);
            }
            else if (val instanceof Function) {
                val = this.toPackedFunc(val);
                stack.tempArgs.push(val);
                stack.storePtr(valueOffset, val._tvmPackedCell.handle);
                stack.storeI32(codeOffset, 10 /* TVMPackedFuncHandle */);
            }
            else if (val instanceof Module) {
                stack.storePtr(valueOffset, val.handle);
                stack.storeI32(codeOffset, 9 /* TVMModuleHandle */);
            }
            else {
                throw new Error("Unsupported argument type " + tp);
            }
        }
    }
    wrapJSFuncAsPackedCFunc(func) {
        const lib = this.lib;
        return (argValues, argCodes, nargs, ret, 
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        _handle) => {
            const jsArgs = [];
            for (let i = 0; i < nargs; ++i) {
                const valuePtr = argValues + i * 8 /* TVMValue */;
                const codePtr = argCodes + i * 4 /* I32 */;
                let tcode = lib.memory.loadI32(codePtr);
                if (tcode == 8 /* TVMObjectHandle */ ||
                    tcode == 14 /* TVMObjectRValueRefArg */ ||
                    tcode == 10 /* TVMPackedFuncHandle */ ||
                    tcode == 9 /* TVMModuleHandle */) {
                    lib.checkCall(lib.exports.TVMCbArgToReturn(valuePtr, codePtr));
                }
                tcode = lib.memory.loadI32(codePtr);
                jsArgs.push(this.retValueToJS(valuePtr, tcode, true));
            }
            const rv = func(...jsArgs);
            if (rv !== undefined && rv !== null) {
                const stack = lib.getOrAllocCallStack();
                const valueOffset = stack.allocRawBytes(8 /* TVMValue */);
                const codeOffset = stack.allocRawBytes(4 /* I32 */);
                this.setPackedArguments(stack, [rv], valueOffset, codeOffset);
                const valuePtr = stack.ptrFromOffset(valueOffset);
                const codePtr = stack.ptrFromOffset(codeOffset);
                stack.commitToWasmMemory();
                lib.checkCall(lib.exports.TVMCFuncSetReturn(ret, valuePtr, codePtr, 1));
                lib.recycleCallStack(stack);
            }
            return 0;
        };
    }
    makePackedFunc(handle) {
        const cell = new PackedFuncCell(handle, this.lib);
        const packedFunc = (...args) => {
            const stack = this.lib.getOrAllocCallStack();
            const valueOffset = stack.allocRawBytes(8 /* TVMValue */ * args.length);
            const tcodeOffset = stack.allocRawBytes(4 /* I32 */ * args.length);
            this.setPackedArguments(stack, args, valueOffset, tcodeOffset);
            const rvalueOffset = stack.allocRawBytes(8 /* TVMValue */);
            const rcodeOffset = stack.allocRawBytes(4 /* I32 */);
            const rvaluePtr = stack.ptrFromOffset(rvalueOffset);
            const rcodePtr = stack.ptrFromOffset(rcodeOffset);
            // commit to wasm memory, till rvalueOffset (the return value don't need to be committed)
            stack.commitToWasmMemory(rvalueOffset);
            this.lib.checkCall(this.exports.TVMFuncCall(handle, stack.ptrFromOffset(valueOffset), stack.ptrFromOffset(tcodeOffset), args.length, rvaluePtr, rcodePtr));
            const ret = this.retValueToJS(rvaluePtr, this.memory.loadI32(rcodePtr), false);
            this.lib.recycleCallStack(stack);
            return ret;
        };
        // Attach attributes to the function type.
        // This is because javascript do not allow us to overload call.
        const ret = packedFunc;
        ret.dispose = () => {
            cell.dispose();
        };
        ret._tvmPackedCell = cell;
        return ret;
    }
    retValueToJS(rvaluePtr, tcode, callbackArg) {
        switch (tcode) {
            case 0 /* Int */:
            case 1 /* UInt */:
                return this.memory.loadI64(rvaluePtr);
            case 2 /* Float */:
                return this.memory.loadF64(rvaluePtr);
            case 3 /* TVMOpaqueHandle */: {
                return this.memory.loadPointer(rvaluePtr);
            }
            case 13 /* TVMNDArrayHandle */: {
                return new NDArray(this.memory.loadPointer(rvaluePtr), false, this.lib);
            }
            case 7 /* TVMDLTensorHandle */: {
                support_1.assert(callbackArg);
                return new NDArray(this.memory.loadPointer(rvaluePtr), true, this.lib);
            }
            case 10 /* TVMPackedFuncHandle */: {
                return this.makePackedFunc(this.memory.loadPointer(rvaluePtr));
            }
            case 9 /* TVMModuleHandle */: {
                return new Module(this.memory.loadPointer(rvaluePtr), this.lib, (ptr) => {
                    return this.makePackedFunc(ptr);
                });
            }
            case 4 /* Null */: return undefined;
            case 6 /* TVMContext */: {
                const deviceType = this.memory.loadI32(rvaluePtr);
                const deviceId = this.memory.loadI32(rvaluePtr + 4 /* I32 */);
                return this.context(deviceType, deviceId);
            }
            case 11 /* TVMStr */: {
                const ret = this.memory.loadCString(this.memory.loadPointer(rvaluePtr));
                return ret;
            }
            case 12 /* TVMBytes */: {
                return this.memory.loadTVMBytes(this.memory.loadPointer(rvaluePtr));
            }
            default:
                throw new Error("Unsupported return type code=" + tcode);
        }
    }
}
exports.Instance = Instance;
/**
 * Asynchrously instantiate a new {@link Instance}.
 *
 * importObject can also be a {@link LibraryProvider} object,
 * a WASI object, or an object containing wasmLibraryProvider field.
 * We can take benefit of syslib implementations from the Emscripten
 * by passing its generated js Module as the imports.
 *
 * @param bufferSource The source to be compiled.
 * @param importObject The import objects.
 * @param logger The system logger.
 */
function instantiate(bufferSource, importObject = {}, logger = console.log) {
    const env = new environment_1.Environment(importObject, logger);
    return WebAssembly.instantiate(bufferSource, env.imports).then((result) => {
        return new Instance(result.module, {}, result.instance, env);
    });
}
exports.instantiate = instantiate;
//# sourceMappingURL=runtime.js.map