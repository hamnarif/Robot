#include <math.h>  // Include the math library for sin and cos functions
# include<Servo.h> // Include Servo library
Servo myServo;  // Create a servo object

 // link lengths
  double lx = 20;  // offser in x direction
  double l1=114;
  double l2=265;
  double l3=218;
  double l4=130;
  //gear ratios
  double g1 = 40;
  double g2 = 50;
  double g3 = 50;
  double g4 = 300;
  double g5 = 9;
  double g6 = 25;

double xg,yg,zg,yaw,pitch,roll;
// Define pin connections
const int dirPin1 = 48;
const int stepPin1 = 49;
const int dirPin2 = 27;
const int stepPin2 = 26;
const int dirPin3 = 53;
const int stepPin3 = 52;
const int dirPin4 = 35; 
const int stepPin4 = 34;
const int dirPin5 = 39;
const int stepPin5 = 38;
const int dirPin6 = 43;
const int stepPin6 = 42;
double thetaI[6],thetaC[6];  // theta_implement & theta_current
double thetaP[6] = {0,0,0,0,0,0};  // theta_preceeding

void IK(double x, double y, double z, double yaw, double pitch, double roll, double* theta);    // inverse kinematics function declaration
void rotateStepper(double steps, int dirPin, int stepPin);  // stepper motor rotation function declaration
void motion(double* theta);                                 // motion function definition
 
const int stepsPerRevolution = 200; // Example for a 1.8° step motor

void setup() {
  pinMode(dirPin1, OUTPUT);
  pinMode(stepPin1, OUTPUT);
  pinMode(dirPin2, OUTPUT);
  pinMode(stepPin2, OUTPUT);
  pinMode(dirPin3, OUTPUT);
  pinMode(stepPin3, OUTPUT);
  pinMode(dirPin4, OUTPUT);
  pinMode(stepPin4, OUTPUT);
  pinMode(dirPin5, OUTPUT);
  pinMode(stepPin5, OUTPUT);
  pinMode(dirPin6, OUTPUT);
  pinMode(stepPin6, OUTPUT);

  Serial.begin(9600);
  // rotation to initial position
  rotateStepper(2000, dirPin3, stepPin3);
  delay(1000);
  rotateStepper(-100, dirPin5, stepPin5);
  delay(1000);
  // Enter coordinates
    xi = 200; yi = -250; zi = 50;
    IK(xi, yi, zi, radians(0), radians(180), radians(0), thetaC);
    for (int i=0; i<6; i++) {
      thetaI[i] = thetaC[i] - thetaP[i];         // subtracting current angles from preceeding to calculate the angles to implement
    }
    double steps1 = (thetaI[0] * stepsPerRevolution) / 360;
    double steps2 = (thetaI[1] * stepsPerRevolution) / 360;
    double steps3 = ((thetaI[2]+90*g3) * stepsPerRevolution) / 360;
    double steps4 = (thetaI[3] * stepsPerRevolution) / 360;
    double steps5 = (thetaI[4] * stepsPerRevolution) / 360;
    double steps6 = (thetaI[5] * stepsPerRevolution) / 360;

    // Rotate each motor
    rotateStepper(steps3, dirPin3, stepPin3);
    delay(1000); // 1 second delay between each motor movement
    rotateStepper(steps4, dirPin4, stepPin4);
    delay(1000);
    rotateStepper(steps5, dirPin5, stepPin5);
    delay(1000);
    rotateStepper(steps6, dirPin6, stepPin6);
    delay(1000);
    rotateStepper(steps1, dirPin1, stepPin1);
    delay(1000);
// Jaw Opening
    myServo.attach(11);  // Attach the servo to pin 11
    myServo.write(0);
    rotateStepper(steps2, dirPin2, stepPin2);
    delay(3000);

    // Jaw closing
    myServo.write(90);
    delay(1000);  // Wait for 1 second

}

void loop() {
  for (int i=0; i<6; i++){
    thetaP[i] = thetaC[i];   // equate preceeding angles to the current of the previous angles
  }
    // Enter coordinates
    xf = 200; yf = 250; zf = 50;
   
    IK(xf, yf, zf, radians(0), radians(180), radians(0), thetaC);
    
    for (int i = 0; i < 6; i++) {
        thetaI[i] = thetaC[i] - thetaP[i];}
        
        motion(thetaI);    // function call to move the manipulator

    xg = -200; yg = -50;   // condition to break the loop
    if (xg<0 or zg<0){   // break the loop after homing
        thetaC[0] = -0*g3; thetaC[1] = 0*g2; thetaC[2] = -160*g3; thetaC[3] = -0*g4; thetaC[4] = 20*g5; thetaC[5] = -0*g6;
        for (int i = 0; i < 6; i++) {
        thetaI[i] = thetaC[i] - thetaP[i];}
        
        motion(thetaI);    // function call to move the manipulator
        return;
    }

}
//Inverse Kinematics function definition
void IK(double xg, double yg, double zg, double yaw, double pitch, double roll, double* theta){

//conditional statement to compensate for y axis offset
if (yg<0){     
  yg = yg-90;
}else{
  yg =yg+50; 
}
// Rotation matrix around the X-axis (Roll)
 double Rx[3][3] = {
    {1, 0, 0},
    {0, cos(roll), -sin(roll)},
    {0, sin(roll), cos(roll)}
  };

  // Rotation matrix around the Y-axis (Pitch)
  double Ry[3][3] = {
    {cos(pitch), 0, sin(pitch)},
    {0, 1, 0},
    {-sin(pitch), 0, cos(pitch)}
  };

  // Rotation matrix around the Z-axis (Yaw)
 double Rz[3][3] = {
  {cos(yaw), -sin(yaw), 0},
  {sin(yaw), cos(yaw), 0},
  {0, 0, 1}
 };

  // Result matrices for intermediate and final results
  double RxRy[3][3];
  double R60[3][3];

  // Multiply Rx and Ry
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      RxRy[i][j] = 0;
      for (int k = 0; k < 3; k++) {
        RxRy[i][j] += Rx[i][k] *Ry[k][j];
      }
    }
  }

  // Multiply the result with Rz
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      R60[i][j] = 0;
      for (int k = 0; k < 3; k++) {
        R60[i][j] += RxRy[i][k] * Rz[k][j];
      }
    }
  }
  double Pwx = xg - l4*R60[0][2];
  double Pwy= yg - l4*R60[1][2];
  double Pwz = zg - l4*R60[2][2];
  double Pw[3] = {Pwx,Pwy,Pwz};

   // Inverse Positioning
  double theta1 = atan2(Pw[1], (Pw[0]+lx));  // angle for joint1

  double C3 = (pow((Pw[0]+lx), 2) + pow(Pw[1], 2) + pow((Pw[2] - l1), 2) - pow(l2, 2) - pow(l3, 2)) / (2 * l2 * l3);  // cosine of theta3
  double S3 = sqrt(1 - pow(C3, 2));  // sine of theta3
  double theta3 = atan2(S3, C3);

  // From squaring x, y, z, terms and solving for theta2
  double A = pow((Pw[0]+lx), 2) + pow(Pw[1], 2) + pow(Pw[2], 2) - pow(l1, 2) - pow(l2, 2) - pow(l3, 2) - 2 * l2 * l3 * C3;
  double alpha = 2 * l1 * l3 * C3;
  double beta = 2 * l1 * l3 * S3;
  double gamma = 2 * l1 * l2;
  double a2 = pow((alpha + gamma), 2) + pow(beta, 2);
  double b2 = -2 * A * (alpha + gamma);
  double c2 = pow(A, 2) - pow(beta, 2);

  double C2 = (-b2 + sqrt(pow(b2, 2) - 4 * a2 * c2)) / (2 * a2);  // cosine of theta2
  double S2 = sqrt(1 - pow(C2, 2));  // sine of theta2
 double theta2 = atan2(S2, C2);

  //Inverse Orientation
double  R30[3][3] = {{cos(theta1)*cos(theta2 - PI/2)*cos(theta3 + PI/2)-cos(theta1)*sin(theta2 - PI/2)*sin(theta3 + PI/2), -cos(theta1)*cos(theta2 - PI/2)*sin(theta3 + PI/2) - cos(theta1)*cos(theta3 + PI/2)*sin(theta2 - PI/2), -sin(theta1)},
{cos(theta2 - PI/2)*cos(theta3 + PI/2)*sin(theta1) - sin(theta1)*sin(theta2 - PI/2)*sin(theta3 + PI/2), -cos(theta2 - PI/2)*sin(theta1)*sin(theta3 + PI/2) - cos(theta3 +PI/2)*sin(theta1)*sin(theta2 - PI/2),  cos(theta1)},
{-cos(theta2 - PI/2)*sin(theta3 + PI/2) -cos(theta3 + PI/2)*sin(theta2 - PI/2),sin(theta2 - PI/2)*sin(theta3 + PI/2) -cos(theta2 - PI/2)*cos(theta3 + PI/2),        0}};
double R30T[3][3] = {
  {R30[0][0],R30[1][0],R30[2][0]}
  ,{R30[0][1],R30[1][1],R30[2][1]}
  ,{R30[0][2],R30[1][2],R30[2][2]}
};
double R63[3][3];
 // Multiply R30T and R60
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      R63[i][j] = 0;
      for (int k = 0; k < 3; k++) {
        R63[i][j] += R30T[i][k] *R60[k][j];
      }
    }
  }
  //calculation for theta5
 double C5 = -R63[1][2]; // Cosine of theta5
 double S5 = sqrt(pow(R63[2][2],2) + pow(R63[0][2],2)); // Sine of theta5
 double theta5 = atan2(S5, C5);

 //computing theta 6 by comparing values in R63
double S6 = -R63[1][1]/S5;
double C6 = R63[1][0]/S5;
double theta6 = atan2(S6,C6);

//computing theta4 by comparing values in R63
double S4 = R63[2][2]/S5;
double C4 = R63[0][2]/S5;
double theta4 = atan2(S4,C4); 
// converting to degrees and scaling for reduction
theta1 = -(theta1) *g1*180/PI;
theta2 = theta2 *g2*180/PI;
theta3 = -(theta3*180/PI)*g3;
theta4 = (-theta4) *g4*180/PI;
theta5 = theta5 *g5*180/PI;
theta6 = -(theta6) *g6*180/PI;

theta[0] = theta1;
theta[1] = theta2;
theta[2] = theta3;
theta[3] = theta4;
theta[4] = theta5;
theta[5] = theta6;
}

// function to rotate stepper motor
void rotateStepper(double steps, const int dirPin, const int stepPin) {
  // Determine direction
  if (steps > 0) {
    digitalWrite(dirPin, HIGH);
  } else {
    digitalWrite(dirPin, LOW);
  }

  // Convert steps to positive
  steps = abs(steps);

  // Step the motor
  for (double i = 0; i < steps; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(1000); // Adjust if necessary for your motor
    digitalWrite(stepPin, LOW);
    delayMicroseconds(1000); // Adjust if necessary for your motor
  }

  // Indicate completion
  Serial.println("Rotation Complete");
}
// motion function definition
void motion(double* theta){
double steps1 = (theta[0] * stepsPerRevolution) / 360;
    double steps2 = (theta[1] * stepsPerRevolution) / 360;
    double steps3 = ((theta[2]) * stepsPerRevolution) / 360;
    double steps4 = (theta[3] * stepsPerRevolution) / 360;
    double steps5 = (theta[4] * stepsPerRevolution) / 360;
    double steps6 = (theta[5] * stepsPerRevolution) / 360;

    rotateStepper(-600, dirPin2,stepPin2);    // adding clearance
    delay(1000);                   
    // Rotate each motor
    rotateStepper(steps3, dirPin3, stepPin3);
    delay(1000); // 1 second delay between each motor movement
    rotateStepper(steps4, dirPin4, stepPin4);
    delay(1000);
    rotateStepper(steps5, dirPin5, stepPin5);
    delay(1000);
    rotateStepper(steps6, dirPin6, stepPin6);
    delay(1000);
    rotateStepper(steps1, dirPin1, stepPin1);
    delay(1000);
    rotateStepper(600, dirPin2,stepPin2);    // removing clearance
    delay(1000); 
    // Jaw Opening
    myServo.attach(11);  // Attach the servo to pin 11
    myServo.write(0);
    rotateStepper(steps2, dirPin2, stepPin2);
    delay(3000);

    // Jaw closing
    myServo.write(90);
    delay(1000);  // Wait for 1 second
}
