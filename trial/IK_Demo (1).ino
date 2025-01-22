#include <Servo.h>
#include <math.h>

// Constants
const int servoPin = 11;
const int stepperPins[3][2] = { {8, 9}, {6, 7}, {4, 5} }; // IN1-IN2 pairs for 3 stepper motors
const int stepsPerRevolution = 200; // Steps per full rotation
const double pi = 3.141592653589793;

// Arm dimensions (length of each segment in units)
const double g1 = 5.0, g2 = 6.0, g3 = 3.0;

// Variables
Servo gripper;
double currentAngles[3] = {0.0, 0.0, 0.0}; // Current angles for the arm (in radians)

void setup() {
    // Initialize the servo for the gripper
    gripper.attach(servoPin);
    gripper.write(90); // Start with the gripper open

    // Initialize the stepper motor pins
    for (int i = 0; i < 3; i++) {
        pinMode(stepperPins[i][0], OUTPUT);
        pinMode(stepperPins[i][1], OUTPUT);
    }

    // Serial monitor for debugging
    Serial.begin(9600);
}

void loop() {
    // Define positions: origin, pickup, and drop-off
    double origin[3] = {5.0, 5.0, 0.0};  // Origin position
    double pickup[3] = {6.0, 6.0, 0.0};  // Pickup position
    double dropoff[3] = {7.0, 7.0, 0.0}; // Drop-off position

    // Move to the pickup position
    moveToPosition(pickup);
    closeGripper(); // Grab the object
    delay(1000);

    // Move to the drop-off position
    moveToPosition(dropoff);
    openGripper(); // Release the object
    delay(1000);

    // Return to the origin position
    moveToPosition(origin);
    delay(1000);
}

void moveToPosition(double target[3]) {
    double targetAngles[3] = {0.0, 0.0, 0.0};

    // Calculate the target angles using inverse kinematics
    calculateIK(target[0], target[1], target[2], targetAngles);

    // Move each motor to the calculated angles
    moveSteppers(currentAngles, targetAngles);

    // Update the current angles
    for (int i = 0; i < 3; i++) {
        currentAngles[i] = targetAngles[i];
    }
}

void calculateIK(double x, double y, double z, double* angles) {
    // Inverse Kinematics calculations
    double r = sqrt(x * x + y * y);
    double R3 = sqrt(r * r + z * z);
    angles[0] = atan2(y, x); // Base rotation

    double C3 = (R3 * R3 - g1 * g1 - g2 * g2) / (2 * g1 * g2);
    C3 = constrain(C3, -1.0, 1.0); // Clamp value to [-1, 1]
    double S3 = sqrt(1 - C3 * C3);

    angles[2] = atan2(S3, C3); // Elbow joint
    double phi = atan2(z, r);
    double beta = atan2(g2 * S3, g1 + g2 * C3);
    angles[1] = phi - beta; // Shoulder joint
}

void moveSteppers(double* currentAngles, double* targetAngles) {
    for (int i = 0; i < 3; i++) {
        int currentSteps = (currentAngles[i] / (2 * pi)) * stepsPerRevolution;
        int targetSteps = (targetAngles[i] / (2 * pi)) * stepsPerRevolution;

        int stepsToMove = targetSteps - currentSteps;
        int direction = (stepsToMove > 0) ? HIGH : LOW;

        for (int j = 0; j < abs(stepsToMove); j++) {
            digitalWrite(stepperPins[i][0], direction);
            digitalWrite(stepperPins[i][1], !direction);
            delayMicroseconds(1000); // Adjust for speed
        }
    }
}

void openGripper() {
    gripper.write(90); // Open the gripper
    Serial.println("Gripper opened");
}

void closeGripper() {
    gripper.write(0); // Close the gripper
    Serial.println("Gripper closed");
}
