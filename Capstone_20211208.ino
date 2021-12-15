int motorPin = 13;      // Motor connected pin
int LEDPin = 12;
int inByte = -1;

void setup() {
  // put your setup code here, to run once:
  Serial.setTimeout(1);
  Serial.begin(9600);
  while (!Serial) {
  ; // wait for serial port to connect. Needed for native USB port only
  }

  pinMode(motorPin, OUTPUT);  // sets the pin as output
  pinMode(LEDPin, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0) {
    // get incoming byte:
    inByte = Serial.readString().toInt();

    
    if (inByte == 0) { // No issue with breathing 48 == 0 
      digitalWrite(LED_BUILTIN, LOW);
      digitalWrite(motorPin, LOW);
      digitalWrite(LEDPin, LOW);
    }
    
    if (inByte == 1) { // When chest breathing is detected single long buzz 49 == 0
      digitalWrite(LED_BUILTIN, HIGH);
      digitalWrite(motorPin, HIGH);
      digitalWrite(LEDPin, HIGH);
      delay(1000);
      digitalWrite(LED_BUILTIN, LOW);
      digitalWrite(motorPin, LOW);
      digitalWrite(LEDPin, LOW);
    }
  
    if (inByte == 2) { // When breathing is too fast 50 == 0
      for (int i = 0; i < 3; i++) {
        digitalWrite(motorPin, HIGH);
        digitalWrite(LED_BUILTIN, HIGH);
        digitalWrite(LEDPin, HIGH);
        delay(250);
        digitalWrite(motorPin, LOW);
        digitalWrite(LED_BUILTIN, LOW);   
        digitalWrite(LEDPin, LOW);     
        delay(250);
      }
    }
  }
}
