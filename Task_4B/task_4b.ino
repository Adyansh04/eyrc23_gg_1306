#include <Arduino.h>
#include <driver/ledc.h>

const byte dir1 = 16;   // Connect DIR1 of motor driver to pin 13
const byte dir2 = 4; 
const byte dir3 = 15; 
const byte dir4 = 2;   // Connect DIR2 of motor driver to pin 12
const byte left = 12;   // Connect PWM1 of motor driver to pin 11
const byte right = 13;   // Connect PWM2 of motor driver to pin 10
const byte ledPin = 21; // LED pin
const byte buzzerPin = 22; // Buzzer pin

int sen1 = 36;
int sen2 = 39;
int sen3 = 34;
int sen4 = 35;
int sen5 = 32;

int maxSpeed = 180;
int rerror = 15;
int inSpeed = 10, outSpeed = 10;

void setup() {
  // Setup pin modes
  Serial.begin(9600);
  pinMode(dir1, OUTPUT);
  pinMode(dir2, OUTPUT);
  pinMode(dir3, OUTPUT);
  pinMode(dir4, OUTPUT);
  pinMode(left, OUTPUT);
  pinMode(right, OUTPUT);
  pinMode(sen1, INPUT);
  pinMode(sen2, INPUT);
  pinMode(sen3, INPUT);
  pinMode(sen4, INPUT);
  pinMode(sen5, INPUT);
  pinMode(ledPin, OUTPUT);
  pinMode(buzzerPin, OUTPUT);

  // Initialize motor driver pins
  digitalWrite(left, LOW);
  digitalWrite(right, LOW);
  digitalWrite(dir1, HIGH);
  digitalWrite(dir2, LOW);
  digitalWrite(dir3, HIGH);
  digitalWrite(dir4, LOW);

  // Initialize LED and buzzer
  // delay(5000);
  digitalWrite(ledPin, HIGH);
  digitalWrite(buzzerPin, HIGH); // Turn on the external buzzer
  delay(5000);
  digitalWrite(ledPin, LOW);
  digitalWrite(buzzerPin,LOW); // Turn off the external buzzer

  ledcSetup(0, 20000, 8);  // Timer 0, frequency 15 kHz, resolution 8-bit
  ledcAttachPin(left, 0); // Attach LEDC channel 0 to pwm1 pin
  ledcAttachPin(right, 1); // Attach LEDC channel 1 to pwm2 pin
}

int junctionCount = 0;

void loop() {
  int val1 = digitalRead(sen1);
  int val2 = digitalRead(sen2);
  int val3 = digitalRead(sen3);
  int val4 = digitalRead(sen4);
  int val5 = digitalRead(sen5);

  Serial.print(val1);
  Serial.print(val2);
  Serial.print(val3);
  Serial.print(val4);
  Serial.println(val5);

  // Motor Control Logic
  if (val3) {
    ledcWrite(0, maxSpeed);      // left
    ledcWrite(1, maxSpeed-rerror);      // right
  }
  if (val2) {
    ledcWrite(0, maxSpeed - inSpeed);      // left
    ledcWrite(1, (maxSpeed-rerror) + inSpeed);      // right
  }
  if (val4) {
    ledcWrite(0, maxSpeed + inSpeed);      // left
    ledcWrite(1, (maxSpeed-rerror) - inSpeed);
  }
  if (val1) {
    ledcWrite(0, maxSpeed + outSpeed);      // left
    ledcWrite(1, (maxSpeed-rerror) - outSpeed);
  }
  if (val5) {
    ledcWrite(0, maxSpeed - outSpeed);      // left
    ledcWrite(1, (maxSpeed-rerror) + outSpeed);
  }

  // Junction Detection
  if (val1 == LOW && val2 == HIGH && val3 == HIGH && val4 == LOW && val5 == LOW) {        //01100
    junctionCount++;
    junctionDecision();
    Serial.print(junctionCount);
  } else if (val1 == LOW && val2 == LOW && val3 == HIGH && val4 == HIGH && val5 == LOW) {   //00110
    junctionCount++;
    junctionDecision();
    Serial.print(junctionCount);
  } else if (val1 == HIGH && val2 == HIGH && val3 == HIGH && val4 == HIGH && val5 == HIGH) {    //11111
    junctionCount++;
    junctionDecision();
    Serial.print(junctionCount);
  } else if (val1 == LOW && val2 == HIGH && val3 == HIGH && val4 == HIGH && val5 == HIGH) {  //01111
    junctionCount++;
    junctionDecision();
    Serial.print(junctionCount);
  } else if (val1 == HIGH && val2 == HIGH && val3 == HIGH && val4 == HIGH and val5 == LOW) {   //11110
    junctionCount++;
    junctionDecision();
    Serial.print(junctionCount);
  } else if (val1 == LOW && val2 == HIGH && val3 == HIGH && val4 == HIGH && val5 == LOW) {     //01110
    junctionCount++;
    junctionDecision();
    Serial.print(junctionCount);
  }
}

void junctionDecision() {
  switch (junctionCount) {
    case 1:
      junction1();
      break;
    case 2:
    case 7:
    case 9:
      junctionStraight();
      break;
    case 3:
    case 5:
    case 6:
    case 8:
      junctionRight();
      break;
    case 4:
    case 10:
      junctionLeft();
      break;
    case 11:
      junctionStop();
      break;
    default:
      break;
  }
}
void junction1(){
  ledcWrite(0,0);
  ledcWrite(1,0);
  digitalWrite(ledPin, HIGH);
  digitalWrite(buzzerPin, HIGH);
  delay(1000);
  digitalWrite(ledPin, LOW);
  digitalWrite(buzzerPin, LOW);
  ledcWrite(0,175);
  ledcWrite(1,maxSpeed - 5);
  delay(1000);
  while(!digitalRead(sen4));
}
void junctionStraight() {
  // Code for straight turn
  // For example:
  Serial.println("Straight");
  // stopAtJunction();
  ledcWrite(0,0);
  ledcWrite(1,0);
  digitalWrite(ledPin, HIGH);
  digitalWrite(buzzerPin, HIGH); // Turn on the external buzzer
  delay(1000);
  digitalWrite(ledPin, LOW);
  digitalWrite(buzzerPin, LOW);
  ledcWrite(0,maxSpeed);
  ledcWrite(1,maxSpeed-7);
  // while(!digitalRead(sen3));
  delay(400);
  return; // Turn off the external buzzer

}

void junctionLeft() {
  // Code for left turn
  // For example:
  Serial.println("Left");
  ledcWrite(0,0);
  ledcWrite(1,0);  // stopAtJunction();
  digitalWrite(ledPin, HIGH);
  digitalWrite(buzzerPin, HIGH); // Turn on the external buzzer
  delay(1000);
  digitalWrite(ledPin, LOW);
  digitalWrite(buzzerPin, LOW); // Turn off the external buzzer

  ledcWrite(0, maxSpeed);
  ledcWrite(1, maxSpeed - 12);
  delay(400);
  ledcWrite(0, 0);      // left
  ledcWrite(1, 175);
  delay(1000);
  while (!digitalRead(sen4));

}

void junctionRight() {
  // Code for right turn
  // For example:
  Serial.println("Right");
  // stopAtJunction();
  ledcWrite(0,0);
  ledcWrite(1,0);

  digitalWrite(ledPin, HIGH);
  digitalWrite(buzzerPin, HIGH); // Turn on the external buzzer
  delay(1000);
  digitalWrite(ledPin, LOW);
  digitalWrite(buzzerPin, LOW); // Turn off the external buzzer

  ledcWrite(0, maxSpeed);
  ledcWrite(1, maxSpeed - rerror);
  delay(300);
  ledcWrite(0, 180);      // left
  ledcWrite(1, 0);
  delay(1000);
  while (!digitalRead(sen2));
  
}

void junctionStop() {
  // Code for stopping at junction
  // For example:
  Serial.println("Stop");
  // stopAtJunction();
  ledcWrite(0,0);
  ledcWrite(1,0);
  delay(1000);
  ledcWrite(0,maxSpeed +5 );
  ledcWrite(1,170);
  delay(3000);
  digitalWrite(ledPin, HIGH);
  digitalWrite(buzzerPin, HIGH);
  ledcWrite(0,0);
  ledcWrite(1,0); // Turn on the external buzzer
  delay(10000); // Additional delay at the junction

}
