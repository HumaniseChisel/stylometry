<script lang="ts">
  let userTypedText = $state("");
  let userName = $state("");

  const expectedText = "the quick brown fox jumps over the lazy dog";

  type KeyLog = {
    key: string;
    ascii: number;
    holdTime: number;
    flightTime: number;
    timestamp: number;
  };

  let keyLogs = $state<KeyLog[]>([]);
  let keyDownTime = $state<number | null>(null);
  let lastKeyUpTime = $state<number | null>(null);
  let isLogExpanded = $state(false);

  let fullKeyLogs = $state<KeyLog[]>([]);

  $effect(() => {
    console.log(
      "Full Key Logs:",
      fullKeyLogs[fullKeyLogs.length - 1]?.key || null
    );
  });

  let keyDownTimes = new Map<string, DOMHighResTimeStamp>();

  function handleKeyDown(event: KeyboardEvent) {
    if (event.repeat) return; // Ignore key repeat

    keyDownTimes.set(event.key, performance.now());

    const key = event.key;
    // fullKeyLogs.push({
    //   key: key,
    //   ascii: key.charCodeAt(0),
    //   holdTime: 0,
    //   flightTime: 0,
    //   timestamp: performance.now(),
    // });
    fullKeyLogs = [...fullKeyLogs];

    // keyDownTime = performance.now();
  }

  function handleKeyUp(event: KeyboardEvent) {
    if (keyDownTime === null) return;

    const key = event.key;
    fullKeyLogs.push({
      key: key,
      ascii: key.charCodeAt(0),
      holdTime: 0,
      flightTime: 0,
      timestamp: performance.now(),
    });
    fullKeyLogs = [...fullKeyLogs];

    // keyDownTime = null;
  }

  function handleInput(event: Event) {
    const target = event.target as HTMLTextAreaElement;
    const newValue = target.value;

    // Allow deleting all text (select all + delete) and reset everything
    if (newValue === "") {
      handleReset();
      return;
    }

    // Update the typed text (already validated in handleKeyDown)
    userTypedText = newValue;
  }

  function handleSubmit() {
    console.log("Name:", userName);
    console.log("Typed text:", userTypedText);
    console.log("Key logs:", keyLogs);
    // Add your submit logic here
  }

  function handleReset() {
    userTypedText = "";
    userName = "";
    keyLogs = [];
    fullKeyLogs = [];
    keyDownTime = null;
    lastKeyUpTime = null;
  }
</script>

<div class="flex min-h-screen items-center justify-center p-4">
  <div class="w-full max-w-3xl space-y-6">
    <h1 class="text-4xl font-bold text-center">stylometry</h1>

    <div class="rounded-lg border border-gray-300 bg-gray-50 p-4">
      <p class="text-lg font-mono text-center">
        <span class="text-gray-400"
          >{expectedText.slice(0, userTypedText.length)}</span
        ><span>{expectedText.slice(userTypedText.length)}</span>
      </p>
    </div>

    <textarea
      value={userTypedText}
      oninput={handleInput}
      onkeydown={handleKeyDown}
      onkeyup={handleKeyUp}
      placeholder="Type the text above here..."
      class="w-full rounded-lg border p-4 font-mono text-lg resize-none focus:outline-none {userTypedText.length ===
      expectedText.length
        ? 'border-green-500 focus:border-green-500 focus:ring-2 focus:ring-green-500'
        : 'border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-500'}"
      rows="3"
    ></textarea>

    <div class="flex gap-4">
      <input
        type="text"
        bind:value={userName}
        placeholder="Your name"
        class="flex-1 rounded-lg border border-gray-300 px-4 py-2 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
      <button
        onclick={handleSubmit}
        disabled={userName.trim() === ""}
        class="rounded-lg px-6 py-2 text-white font-semibold focus:outline-none focus:ring-2 focus:ring-offset-2 {userName.trim() ===
        ''
          ? 'bg-blue-400 cursor-not-allowed'
          : 'bg-blue-600 hover:bg-blue-700 focus:ring-blue-500'}"
      >
        Submit
      </button>
      <button
        onclick={handleReset}
        class="rounded-lg bg-gray-600 px-6 py-2 text-white font-semibold hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2"
      >
        Reset
      </button>
    </div>

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
            {#each keyLogs as log, index (log.timestamp)}
              <div class="py-1 border-b border-gray-100 last:border-b-0">
                <span class="text-gray-600">#{index + 1}</span> Key:
                <span class="font-semibold">'{log.key}'</span>
                | ASCII: <span class="font-semibold">{log.ascii}</span> | Hold:
                <span class="text-blue-600">{log.holdTime}ms</span>
                | Flight: <span class="text-green-600">{log.flightTime}ms</span>
              </div>
            {/each}
          {/if}
        </div>
      {/if}
    </div>
  </div>
</div>
