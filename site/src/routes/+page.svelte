<script lang="ts">
  const API_BASE_URL = "http://localhost:8000"; // Base API endpoint
  // const API_BASE_URL = "https://43f409bdcf90.ngrok-free.app"; // Base API endpoint

  let userTypedText = $state("");
  let userName = $state("");

  const expectedText = ".tie5Roanl";

  type KeyLog = {
    key: string;
    keyDownTime: number;
    holdTime: number;
    flightTime: number;
  };

  let fullKeyLogs = $state<KeyLog[]>([]);
  let keyDownTime = $state<number | null>(null);
  let lastKeyUpTime = $state<number | null>(null);
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

  // Debugging logger
  $effect(() => {
    console.log(
      "Full Key Logs:",
      fullKeyLogs[fullKeyLogs.length - 1]?.key || null
    );
  });

  let keyDownTimes = new Map<string, DOMHighResTimeStamp>();

  function handleKeyDown(event: KeyboardEvent) {
    if (event.repeat) return; // Ignore key repeat

    const key = event.key;
    // Basic validation to ensure they type the correct character
    if (key != expectedText.at(userTypedText.length)) {
      event.preventDefault();
      wrongChar = true;
      return;
    }

    wrongChar = false;
    keyDownTimes.set(key, performance.now());
  }

  function handleKeyUp(event: KeyboardEvent) {
    const key = event.key;
    const keyDownTime = keyDownTimes.get(key);
    if (keyDownTime === undefined) return;
    keyDownTimes.delete(key);

    const keyUpTime = performance.now();
    const holdTime = keyUpTime - keyDownTime;

    let flightTime = 0;
    if (fullKeyLogs.length > 0) {
      const last = fullKeyLogs[fullKeyLogs.length - 1];
      const lastKeyUpTime = last.keyDownTime + last.holdTime;
      flightTime = keyDownTime - lastKeyUpTime;
      // if (lastKeyUpTime < keyDownTime) {
      //   flightTime = keyDownTime - lastKeyUpTime;
      // }
    }

    fullKeyLogs.push({
      key,
      keyDownTime,
      holdTime,
      flightTime,
    });
    fullKeyLogs = [...fullKeyLogs];

    // Check for completion
    if (userTypedText === expectedText) {
      handleSubmit();
    }
  }

  function handleInput(event: Event) {
    const target = event.target as HTMLTextAreaElement;
    const newValue = target.value;

    // Allow deleting all text (select all + delete) and reset logs
    if (newValue === "") {
      handleReset();
      return;
    }

    userTypedText = newValue;
  }

  async function handleSubmit() {
    if (isSubmitting) return;
    isSubmitting = true;

    try {
      uploadStatus = null;

      const data = {
        name: userName,
        keystrokeLogs: fullKeyLogs,
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
    userTypedText = "";
    fullKeyLogs = [];
    keyDownTime = null;
    lastKeyUpTime = null;
    // Note: We do NOT reset userName so you can easily do multiple trials
  }
</script>

<div class="flex min-h-screen items-center justify-center p-4">
  <div class="w-full max-w-3xl space-y-6">
    <h1 class="text-4xl font-bold text-center">typometry</h1>
    <h1>* NEGATIVE FLIGHT TIMES ARE NOW ALLOWED *</h1>

    <div>
      <input
        id="username"
        type="text"
        bind:value={userName}
        disabled={isSubmitting}
        placeholder="Enter your name to unlock the test..."
        class="w-full rounded-lg border border-gray-300 px-4 py-2 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 disabled:text-gray-500"
      />
      {numTyped}
    </div>

    <div class="rounded-lg border border-gray-300 bg-gray-50 p-4 select-none">
      <p class="text-lg font-mono text-center">
        <span class="text-gray-400"
          >{expectedText.slice(0, userTypedText.length)}</span
        ><span>{expectedText.slice(userTypedText.length)}</span>
      </p>
    </div>

    <div class="relative">
      <textarea
        value={userTypedText}
        oninput={handleInput}
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
        {userTypedText.length === expectedText.length && !isSubmitting
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

    <div class="flex justify-end">
      <button
        onclick={handleReset}
        disabled={userTypedText.length === 0 || isSubmitting}
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
          {#if fullKeyLogs.length === 0}
            <p class="text-gray-400">No keystrokes recorded yet...</p>
          {:else}
            {#each fullKeyLogs as log, index (log.keyDownTime)}
              <div class="py-1 border-b border-gray-100 last:border-b-0">
                <span class="text-gray-600">#{index + 1}</span> Key:
                <span class="font-semibold">'{log.key}'</span>
                | Hold:
                <span class="text-blue-600">{log.holdTime.toFixed(1)}ms</span>
                | Flight:
                <span class="text-green-600">{log.flightTime.toFixed(1)}ms</span
                >
              </div>
            {/each}
          {/if}
        </div>
      {/if}
    </div>
  </div>
</div>
