/*********
  Rui Santos
  Complete project details at https://randomnerdtutorials.com
*********/

#include <HTTPClient.h>

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include <MAX3010x.h>

#include "filter.h"

#define BLYNK_TEMPLATE_ID "TMPL63tH_EDn1"
#define BLYNK_TEMPLATE_NAME "Alertify"
#define BLYNK_AUTH_TOKEN "AbfqhbZ4A6r9Qo_8ONp6zRy3KYds-WEw"

#include <BlynkSimpleEsp32.h>

#define WIFI_SSID "Biznet"
#define WIFI_PASS "Biznet911"

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 64 // OLED display height, in pixels

float max_acc = NAN;
float min_acc = NAN;

// Declaration for an SSD1306 display connected to I2C (SDA, SCL pins)
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
MAX30105 sensor;
Adafruit_MPU6050 mpu;

const auto kSamplingRate = sensor.SAMPLING_RATE_400SPS;
const float kSamplingFrequency = 400.0;

// Finger Detection Threshold and Cooldown
const unsigned long kFingerThreshold = 10000;
const unsigned int kFingerCooldownMs = 500;

// Edge Detection Threshold (decrease for MAX30100)
const float kEdgeThreshold = -500.0;

// Filters
const float kLowPassCutoff = 5.0;
const float kHighPassCutoff = 0.5;

// Averaging
const bool kEnableAveraging = true;
const int kAveragingSamples = 10;
const int kSampleThreshold = 3;

BlynkTimer timer;
int blynk_timer;
const int blynk_interval = 250;

// Filter Instances
LowPassFilter low_pass_filter_red(kLowPassCutoff, kSamplingFrequency);
LowPassFilter low_pass_filter_ir(kLowPassCutoff, kSamplingFrequency);
HighPassFilter high_pass_filter(kHighPassCutoff, kSamplingFrequency);
Differentiator differentiator(kSamplingFrequency);
MovingAverageFilter<kAveragingSamples> averager_bpm;
MovingAverageFilter<kAveragingSamples> averager_r;
MovingAverageFilter<kAveragingSamples> averager_spo2;

// Statistic for pulse oximetry
MinMaxAvgStatistic stat_red;
MinMaxAvgStatistic stat_ir;

sensors_event_t a, g, temp;

// R value to SpO2 calibration factors
// See https://www.maximintegrated.com/en/design/technical-documents/app-notes/6/6845.html
float kSpO2_A = 1.5958422;
float kSpO2_B = -34.6596622;
float kSpO2_C = 112.6898759;

// Timestamp of the last heartbeat
long last_heartbeat = 0;

// Timestamp for finger detection
long finger_timestamp = 0;
bool finger_detected = false;

// Last diff to detect zero crossing
float last_diff = NAN;
bool crossed = false;
long crossed_time = 0;

int bpm_reading = 0;
float spo2_reading = 0;

const int seconds = 1000;
const int minutes = 60 * seconds;
const int hours = 60 * minutes;
const int days = 24 * hours;

int fall_timer;
const int fall_interval = 10 * seconds;
bool fall_detected = false;

int temp_timer;
const int temp_interval = 5 * minutes;

int spo2_timer;
const int spo2_interval = 5 * minutes;

int bpm_timer;
const int bpm_interval = 5 * minutes;

bool finger_output = false;

float last_acceleration_x = NAN;
float last_acceleration_y = NAN;
float last_acceleration_z = NAN;
float last_acceleration_diff = NAN;

void setup()
{
  timer.setInterval(1000L, []()
                    {
    Blynk.virtualWrite(V1, bpm_reading);
    Blynk.virtualWrite(V2, spo2_reading);
    Blynk.virtualWrite(V3, fall_detected);
    Blynk.virtualWrite(V4, finger_output); });
  Serial.begin(9600);

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C))
  { // Address 0x3D for 128x64
    Serial.println(F("SSD1306 allocation failed"));
    for (;;)
      ;
  }
  display.clearDisplay();

  // Initialized OLED Display
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 10);
  display.println("Display initialized");
  display.display();
  delay(2000);

  display.clearDisplay();
  display.setCursor(0, 10);
  display.println("Initializing MPU6050");
  display.display();

  if (!mpu.begin())
  {
    display.setCursor(0, 20);
    display.println("Failed to find MPU6050 chip");
    display.display();
    while (1)
    {
      delay(10);
    }
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  display.clearDisplay();
  display.setCursor(0, 10);
  display.println("MPU6050 Initialized");
  display.display();
  delay(2000);

  display.clearDisplay();
  display.setCursor(0, 10);
  display.println("Initializing MAX30102");
  display.display();

  if (sensor.begin() && sensor.setSamplingRate(kSamplingRate))
  {
    display.clearDisplay();
    display.setCursor(0, 10);
    display.println("MAX30102 Initialized");
    display.display();
    delay(2000);
  }
  else
  {
    display.setCursor(0, 20);
    display.println("Failed to find MAX30102 chip");
    display.display();
    while (1)
      ;
  }

  display.clearDisplay();

  display.println("Initializing Blynk\n");
  display.display();
  Blynk.begin(BLYNK_AUTH_TOKEN, WIFI_SSID, WIFI_PASS);
  display.println("Blynk initialized\n");
  display.display();

  fall_timer = millis();
  temp_timer = millis();
  spo2_timer = millis();
  bpm_timer = millis();

  blynk_timer = millis();

  display.clearDisplay();
}

void displayReading()
{
  display.setCursor(0, 10);
  display.print("Temp: ");
  display.print(temp.temperature);
  display.println(" C");

  if (finger_output)
  {
    display.print("BPM: ");
    display.println(bpm_reading);
    display.print("SpO2: ");
    display.println(spo2_reading);
  }
  else
  {
    display.println("Finger not detected");
  }

  if (fall_detected)
  {
    if (millis() - fall_timer < fall_interval)
    {
      display.println("Fall detected");
    }
    else
    {
      fall_detected = false;
    }
  }

  display.display();
}

void sendLineNotify(String message)
{
  HTTPClient http;
  http.begin("https://notify-api.line.me/api/notify");
  http.addHeader("Content-Type", "application/x-www-form-urlencoded");
  http.addHeader("Authorization", "Bearer 9rqKxPXzYNCEsrFrwRhDzIL7UgM3ld45dEF7W7KmmLe");
  String payload = "message=" + message;
  int httpResponseCode = http.POST(payload);
  if (httpResponseCode > 0)
  {
    String response = http.getString();
    Serial.println(response);
  }
  else
  {
    Serial.print("Error on sending POST: ");
    Serial.println(httpResponseCode);
  }
  http.end();
}

void loop()
{
  display.clearDisplay();

  mpu.getEvent(&a, &g, &temp);

  auto sample = sensor.readSample(1000);
  float current_value_red = sample.red;
  float current_value_ir = sample.ir;

  // Detect Finger using raw sensor value
  if (sample.red > kFingerThreshold)
  {
    if (millis() - finger_timestamp > kFingerCooldownMs)
    {
      finger_detected = true;
    }
  }
  else
  {
    // Reset values if the finger is removed
    differentiator.reset();
    averager_bpm.reset();
    averager_r.reset();
    averager_spo2.reset();
    low_pass_filter_red.reset();
    low_pass_filter_ir.reset();
    high_pass_filter.reset();
    stat_red.reset();
    stat_ir.reset();

    finger_detected = false;
    finger_timestamp = millis();
  }

  if (finger_detected)
  {
    current_value_red = low_pass_filter_red.process(current_value_red);
    current_value_ir = low_pass_filter_ir.process(current_value_ir);

    // Statistics for pulse oximetry
    stat_red.process(current_value_red);
    stat_ir.process(current_value_ir);

    // Heart beat detection using value for red LED
    float current_value = high_pass_filter.process(current_value_red);
    float current_diff = differentiator.process(current_value);

    // Valid values?
    if (!isnan(current_diff) && !isnan(last_diff))
    {

      // Detect Heartbeat - Zero-Crossing
      if (last_diff > 0 && current_diff < 0)
      {
        crossed = true;
        crossed_time = millis();
      }

      if (current_diff > 0)
      {
        crossed = false;
      }

      // Detect Heartbeat - Falling Edge Threshold
      if (crossed && current_diff < kEdgeThreshold)
      {
        if (last_heartbeat != 0 && crossed_time - last_heartbeat > 300)
        {
          // Show Results
          int bpm = 60000 / (crossed_time - last_heartbeat);
          float rred = (stat_red.maximum() - stat_red.minimum()) / stat_red.average();
          float rir = (stat_ir.maximum() - stat_ir.minimum()) / stat_ir.average();
          float r = rred / rir;
          float spo2 = kSpO2_A * r * r + kSpO2_B * r + kSpO2_C;

          if (bpm > 50 && bpm < 250)
          {
            // Average?
            if (kEnableAveraging)
            {
              int average_bpm = averager_bpm.process(bpm);
              int average_r = averager_r.process(r);
              int average_spo2 = averager_spo2.process(spo2);

              // Show if enough samples have been collected
              if (averager_bpm.count() >= kSampleThreshold)
              {
                bpm_reading = average_bpm;
                spo2_reading = average_spo2;
              }
            }
            else
            {
              bpm_reading = bpm;
              spo2_reading = spo2;
            }
          }

          // Reset statistic
          stat_red.reset();
          stat_ir.reset();
        }

        crossed = false;
        last_heartbeat = crossed_time;
      }
    }

    last_diff = current_diff;

    finger_output = true;
  }
  else
  {
    finger_output = false;
  }

  if (isnan(last_acceleration_x))
  {
    last_acceleration_x = a.acceleration.x;
    last_acceleration_y = a.acceleration.y;
    last_acceleration_z = a.acceleration.z;
  }

  float acceletarion_diff = abs(a.acceleration.x - last_acceleration_x) + abs(a.acceleration.y - last_acceleration_y) + abs(a.acceleration.z - last_acceleration_z);
  display.println("Acc " + String(acceletarion_diff));

  if (isnan(max_acc) || acceletarion_diff > max_acc)
  {
    max_acc = acceletarion_diff;
  }

  if (isnan(min_acc) || acceletarion_diff < min_acc)
  {
    min_acc = acceletarion_diff;
  }

  Serial.println("Max Acc: " + String(max_acc));
  Serial.println("Min Acc: " + String(min_acc));

  if (!isnan(last_acceleration_diff))
  {
    if (last_acceleration_diff > 100 && millis() - fall_timer > fall_interval && finger_output)
    {
      if (acceletarion_diff < 10)
      {
        fall_timer = millis();
        fall_detected = true;
        sendLineNotify("Fall detected!");
      }
    }
  }
  last_acceleration_diff = acceletarion_diff;


  // if ((acceletarion > 15 || abs(last_acceleration - acceletarion) > 5) && millis() - fall_timer > fall_interval && finger_output)
  // {
  //   fall_timer = millis();
  //   fall_detected = true;
  //   sendLineNotify("Fall detected!");
  // }

  // last_acceleration = acceletarion;

  if (temp.temperature > 38 && millis() - temp_timer > temp_interval && finger_output)
  {
    temp_timer = millis();
    sendLineNotify("High temperature detected: " + String(temp.temperature) + " C");
  }

  if (spo2_reading < 90 && millis() - spo2_timer > spo2_interval && finger_output)
  {
    spo2_timer = millis();
    sendLineNotify("Low SpO2 detected: " + String(spo2_reading) + "%");
  }

  if (((bpm_reading < 60 && bpm_reading > 10) || bpm_reading > 100) && millis() - bpm_timer > bpm_interval && finger_output)
  {
    bpm_timer = millis();
    sendLineNotify("Abnormal heart rate detected: " + String(bpm_reading) + " BPM");
  }

  displayReading();

  if (millis() - blynk_timer > blynk_interval)
  {
    Blynk.run();
    timer.run();
    blynk_timer = millis();

    Serial.println("Max Acc: " + String(max_acc));
    Serial.println("Min Acc: " + String(min_acc));
  }
}
