const apiKey = "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";
const model = "meta/llama-3.1-8b-instruct";
const url = "https://integrate.api.nvidia.com/v1/chat/completions";

async function test() {
  console.log("Starting API Request with Llama 3.1 8B...");
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        model: model,
        messages: [{ role: "user", content: "hello" }],
        temperature: 0.1,
        max_tokens: 128
      })
    });

    console.log("Response Status:", res.status, res.statusText);
    const data = await res.json();
    console.log("Response Data:", JSON.stringify(data, null, 2));
  } catch (err) {
    console.error("API Request Failed:", err);
  }
}

test();
