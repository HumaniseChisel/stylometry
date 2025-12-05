<script lang="ts">
  const API_BASE_URL = "http://localhost:8000"; // Base API endpoint

  let userName = $state("");

  // const expectedText = ".tie5Roanl";
  const expectedText = "the quick brown fox jumps over the lazy dog";

  type KeyLog = {
    key: string;
    keyDownTime: number;
    holdTime?: number;
    flightTime?: number;
  };

  let keyLogs = $state<KeyLog[]>([]);
  let charsTyped = $state(0);
  let keyDownTimes = new Map<string, number>();

  let isLogExpanded = $state(false);
  let wrongChar = $state(false);
  let numTyped = $state(0);

  // New state to lock UI during auto-submit
  let isSubmitting = $state(false);

  let textAreaRef: HTMLTextAreaElement;

  let uploadStatus = $state<{
    type: "success" | "error";
    message: string;
  } | null>(null);

  type InferenceResult = {
    predicted_user: string;
    confidence: number;
    num_keystrokes: number;
    original_filename: string;
    all_classes: Record<string, number>;
  };

  let inferenceResult = $state<InferenceResult | null>(null);
  let isInferring = $state(false);
  let inferenceError = $state<string | null>(null);

  function handleKeyDown(event: KeyboardEvent) {
    event.preventDefault();
    if (event.repeat) return; // Ignore key repeat

    const key = event.key;

    if (key != expectedText.at(charsTyped)) {
      wrongChar = true;
      return;
    }

    wrongChar = false;
    charsTyped++;
    keyDownTimes.set(key, keyLogs.length);
    keyLogs.push({
      key,
      keyDownTime: performance.now(),
    });
  }

  function handleKeyUp(event: KeyboardEvent) {
    const key = event.key;
    const keyLogIdx = keyDownTimes.get(key);
    if (keyLogIdx === undefined) return;

    const keyDownTime = keyLogs[keyLogIdx].keyDownTime;
    if (keyDownTime === undefined) {
      alert("Key down time not found");
      return;
    }
    keyDownTimes.delete(key);

    const keyUpTime = performance.now();
    const holdTime = keyUpTime - keyDownTime;

    let flightTime = 0;
    if (keyLogs.length > 0) {
      const last = keyLogs[keyLogs.length - 1];
      if (last.holdTime !== undefined) {
        const lastKeyUpTime = last.keyDownTime + last.holdTime;
        flightTime = keyDownTime - lastKeyUpTime;
      }
    }

    keyLogs[keyLogIdx] = {
      key,
      keyDownTime,
      holdTime,
      flightTime,
    };

    // Check for completion
    if (keyLogs.length === expectedText.length) {
      const loggedString = keyLogs.map(({ key }) => key).join("");
      if (loggedString !== expectedText) {
        alert(`Shoot, got ${loggedString}`);
      } else {
        handleSubmit();
      }
    }
  }

  async function handleSubmit() {
    if (isSubmitting) return;

    const loggedString = keyLogs.map(({ key }) => key).join("");
    if (loggedString !== expectedText) {
      alert(`Got "${loggedString}" instead of "${expectedText}"`);
      return;
    }

    isSubmitting = true;

    try {
      uploadStatus = null;

      const data = {
        name: userName,
        keystrokeLogs: keyLogs,
      };

      const jsonBlob = new Blob([JSON.stringify(data)], {
        type: "application/json",
      });
      const jsonFile = new File([jsonBlob], "data.json", {
        type: "application/json",
      });

      const formData = new FormData();
      formData.append("file", jsonFile);

      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      uploadStatus = {
        type: "success",
        message: "Great job! Uploading and resetting...",
      };

      handleReset();
      uploadStatus = null;
      isSubmitting = false;
      numTyped += 1;
      setTimeout(() => {
        textAreaRef.focus();
      }, 0);
    } catch (error) {
      isSubmitting = false;
      uploadStatus = {
        type: "error",
        message:
          error instanceof Error
            ? error.message
            : "Upload failed. Please try again.",
      };
    }
  }

  function handleReset() {
    keyLogs = [];
    keyDownTimes = new Map<string, DOMHighResTimeStamp>();
    charsTyped = 0;
  }

  async function handleInference() {
    if (isInferring) return;
    isInferring = true;
    inferenceError = null;
    inferenceResult = null;

    try {
      const data = {
        name: userName,
        keystrokeLogs: keyLogs,
      };

      const jsonBlob = new Blob([JSON.stringify(data)], {
        type: "application/json",
      });
      const jsonFile = new File([jsonBlob], "data.json", {
        type: "application/json",
      });

      const formData = new FormData();
      formData.append("file", jsonFile);

      const response = await fetch(`${API_BASE_URL}/inference`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Inference failed: ${response.statusText}`);
      }

      const result = await response.json();
      inferenceResult = result;
    } catch (error) {
      inferenceError =
        error instanceof Error
          ? error.message
          : "Inference failed. Please try again.";
    } finally {
      isInferring = false;
    }
  }
</script>

<div class="flex min-h-screen items-center justify-center p-4">
  <div class="w-full max-w-3xl space-y-6">
    <h1 class="text-4xl font-bold text-center">typometry</h1>
    <!-- TODO: Why is this here? -->
    <!-- <h1>* NEGATIVE FLIGHT TIMES ARE NOW ALLOWED *</h1> -->

    <div>
      <input
        id="username"
        type="text"
        bind:value={userName}
        disabled={isSubmitting}
        placeholder="Enter your name to unlock the test..."
        class="w-full rounded-lg border border-gray-300 px-4 py-2 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 disabled:text-gray-500"
      />
      <!-- TODO: idek what to do with this -->
      <!-- {numTyped} -->
    </div>

    <div class="rounded-lg border border-gray-300 bg-gray-50 p-4 select-none">
      <p class="text-lg font-mono">
        <span class="text-gray-400">{expectedText.slice(0, charsTyped)}</span
        ><span>{expectedText.slice(charsTyped)}</span>
      </p>
    </div>

    <div class="relative">
      <textarea
        value={expectedText.slice(0, charsTyped)}
        onkeydown={handleKeyDown}
        onkeyup={handleKeyUp}
        disabled={userName.trim() === "" || isSubmitting}
        placeholder={userName.trim() === ""
          ? "Please enter your name above first."
          : "Type the text exactly as shown above..."}
        class="w-full rounded-lg border p-4 font-mono text-lg resize-none focus:outline-none transition-all
        {userName.trim() === '' || isSubmitting
          ? 'bg-gray-100 text-gray-400 cursor-not-allowed border-gray-200'
          : 'bg-white text-black border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-500'}
        {keyLogs.length === expectedText.length && !isSubmitting
          ? 'border-green-500 focus:border-green-500 focus:ring-green-500'
          : ''}
        {wrongChar &&
          'focus:border-red-500 focus:ring-2 focus:ring-red-500 shake'}"
        rows="3"
        bind:this={textAreaRef}
      ></textarea>

      {#if isSubmitting}
        <div
          class="absolute inset-0 flex items-center justify-center bg-white/50 backdrop-blur-sm rounded-lg"
        >
          <span class="font-semibold text-blue-600 animate-pulse"
            >Processing...</span
          >
        </div>
      {/if}
    </div>

    <div class="flex justify-end gap-3">
      <button
        onclick={handleSubmit}
        disabled={keyLogs.length === 0 || isSubmitting || isInferring}
        class="rounded-lg bg-green-600 px-6 py-2 text-white font-semibold hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isSubmitting ? "Submitting..." : "Submit"}
      </button>
      <button
        onclick={handleInference}
        disabled={keyLogs.length === 0 || isInferring || isSubmitting}
        class="rounded-lg bg-blue-600 px-6 py-2 text-white font-semibold hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isInferring ? "Inferring..." : "Inference"}
      </button>
      <button
        onclick={handleReset}
        disabled={keyLogs.length === 0 || isSubmitting}
        class="rounded-lg bg-gray-600 px-6 py-2 text-white font-semibold hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Reset
      </button>
    </div>

    {#if uploadStatus}
      <div
        class="rounded-lg p-4 text-center font-medium transition-all {uploadStatus.type ===
        'success'
          ? 'bg-green-100 text-green-800 border border-green-300'
          : 'bg-red-100 text-red-800 border border-red-300'}"
      >
        {uploadStatus.message}
      </div>
    {/if}

    <div class="rounded-lg border border-gray-300 bg-gray-50 p-4">
      <button
        onclick={() => (isLogExpanded = !isLogExpanded)}
        class="w-full flex items-center justify-between text-lg font-semibold hover:text-blue-600 transition-colors"
      >
        <span>Keystroke Log</span>
        <span class="text-2xl">{isLogExpanded ? "−" : "+"}</span>
      </button>
      {#if isLogExpanded}
        <div
          class="mt-2 max-h-64 overflow-y-auto bg-white rounded border border-gray-200 p-2 font-mono text-sm"
        >
          {#if keyLogs.length === 0}
            <p class="text-gray-400">No keystrokes recorded yet...</p>
          {:else}
            {#each keyLogs as log, index (log.keyDownTime)}
              <div class="py-1 border-b border-gray-100 last:border-b-0">
                <span class="text-gray-600">#{index + 1}</span> Key:
                <span class="font-semibold">'{log.key}'</span>
                | Hold:
                <span class="text-blue-600"
                  >{log.holdTime?.toFixed(1) ?? "N/A"}ms</span
                >
                | Flight:
                <span class="text-green-600"
                  >{log.flightTime?.toFixed(1) ?? "N/A"}ms</span
                >
              </div>
            {/each}
          {/if}
        </div>
      {/if}
    </div>

    {#if inferenceResult || inferenceError}
      <div class="rounded-lg border border-gray-300 bg-gray-50 p-4">
        <h3 class="text-lg font-semibold mb-3">Inference Results</h3>
        {#if inferenceError}
          <div
            class="bg-red-100 text-red-800 border border-red-300 rounded-lg p-4"
          >
            <p class="font-medium">Error: {inferenceError}</p>
          </div>
        {:else if inferenceResult}
          <div class="bg-white rounded border border-gray-200 p-4 space-y-3">
            <div
              class="flex justify-between items-center pb-2 border-b border-gray-200"
            >
              <span class="font-semibold text-gray-700">Predicted User:</span>
              <span class="text-xl font-bold text-blue-600"
                >{inferenceResult.predicted_user}</span
              >
            </div>
            <div
              class="flex justify-between items-center pb-2 border-b border-gray-200"
            >
              <span class="font-semibold text-gray-700">Confidence:</span>
              <span class="text-lg font-semibold text-green-600"
                >{(inferenceResult.confidence * 100).toFixed(2)}%</span
              >
            </div>
            <div class="flex justify-between items-center">
              <span class="font-semibold text-gray-700"
                >Keystrokes Analyzed:</span
              >
              <span class="text-lg font-semibold"
                >{inferenceResult.num_keystrokes}</span
              >
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>
