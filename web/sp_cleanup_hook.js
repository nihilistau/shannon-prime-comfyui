/**
 * Shannon-Prime: Cancel/interrupt cleanup hook.
 *
 * When the user cancels or an error interrupts generation, call /sp/cleanup
 * to unload all VRAM-resident models and clear the CUDA allocator.
 * This frees the GPU immediately rather than waiting for the next generation
 * to start (which is when ComfyUI would normally evict the old model).
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

async function spCleanup(reason) {
    try {
        const r = await api.fetchApi("/sp/cleanup", { method: "POST" });
        if (r.ok) {
            console.log(`[Shannon-Prime] VRAM cleared on ${reason}`);
        }
    } catch (e) {
        console.warn("[Shannon-Prime] /sp/cleanup call failed:", e);
    }
}

app.registerExtension({
    name: "ShannonPrime.CancelCleanup",
    setup() {
        // Fires when the queue is interrupted by the user pressing cancel
        api.addEventListener("execution_interrupted", () => spCleanup("cancel"));
        // Fires on node execution error (model left loaded but idle)
        api.addEventListener("execution_error",       () => spCleanup("error"));
    }
});
