#include "FS.h"
#include "LittleFS.h"

// Expose C functions from run.c to C++ compiler
extern "C" {
  void run_llama(const char* model_path, const char* tokenizer_path, float temperature, float topp, int steps, const char* prompt);
}

// Queue handle for passing incoming prompts to the FreeRTOS task
QueueHandle_t promptQueue;

void llamaTask(void *pvParameters) {
  Serial.println("\n--- Starting LittleFS Initialization ---");
  
  if (!LittleFS.begin(false)) {
    Serial.println("❌ LittleFS Mount Failed! Check if filesystem was uploaded.");
    vTaskDelete(NULL);
    return;
  }

  Serial.println("✅ LittleFS Mounted Successfully.");
  
  if (!LittleFS.exists("/stories260K.bin")) {
    Serial.println("❌ Error: /stories260K.bin not found on LittleFS!");
    vTaskDelete(NULL);
    return;
  }
  
  if (!LittleFS.exists("/tok512.bin")) {
    Serial.println("❌ Error: /tok512.bin not found on LittleFS!");
    vTaskDelete(NULL);
    return;
  }

  Serial.println("✅ Model and Tokenizer files located.");
  Serial.println("\n==========================================");
  Serial.println(" Type a prompt in Serial Monitor & press Enter!");
  Serial.println("==========================================\n");

  char receivedPrompt[256];

  // Infinite processing loop waiting for prompts from queue
  for (;;) {
    if (xQueueReceive(promptQueue, &receivedPrompt, portMAX_DELAY) == pdTRUE) {
      Serial.println("\n------------------------------------------");
      Serial.printf("Prompt: \"%s\"\n", receivedPrompt);
      Serial.println("--- Starting Llama Inference (Flash Streaming) ---\n");

      // Execute model generation with dynamic prompt
      run_llama("/stories260K.bin", "/tok512.bin", 0.8f, 0.9f, 250, receivedPrompt);

      Serial.println("\n--- Inference Complete ---");
      Serial.println("\nEnter next prompt:");
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(2000); // Allow hardware serial connection to settle

  Serial.println("==========================================");
  Serial.println("   ESP32-S3 Tiny Llama Inference Engine   ");
  Serial.println("==========================================");

  // Non-blocking memory mode log
  if (ESP.getPsramSize() == 0) {
    Serial.println("ℹ️ PSRAM not detected. Running in Flash-Streaming mode.");
  } else {
    Serial.printf("PSRAM Available: %d Bytes\n", ESP.getFreePsram());
  }

  // Create queue capable of holding up to 5 prompt strings
  promptQueue = xQueueCreate(5, sizeof(char[256]));

  // Create background task on Core 1 dedicated to running LLM
  xTaskCreatePinnedToCore(
    llamaTask,   
    "LlamaTask",  
    24576,       // 24KB Stack allocation (safely fits inside internal SRAM)
    NULL,           
    1,           
    NULL,        
    1            // Pin execution to Core 1
  );

  // Send an initial startup prompt to start inference automatically
  char defaultPrompt[] = "Once upon a time";
  xQueueSend(promptQueue, &defaultPrompt, portMAX_DELAY);
}

void loop() {
  // Read typed text from the Serial Monitor
  if (Serial.available() > 0) {
    String inputString = Serial.readStringUntil('\n');
    inputString.trim(); // Strip carriage returns and spaces

    if (inputString.length() > 0) {
      char promptBuffer[256];
      inputString.toCharArray(promptBuffer, sizeof(promptBuffer));

      // Push user text into queue for llamaTask to consume
      xQueueSend(promptQueue, &promptBuffer, pdMS_TO_TICKS(100));
    }
  }

  vTaskDelay(pdMS_TO_TICKS(50)); // Poll serial smoothly
}