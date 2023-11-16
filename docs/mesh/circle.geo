
// Geometry
SetFactory("OpenCASCADE");
Circle(1) = {0.0,0.0,0.0,1000.0,0.0,2*Pi};
Curve Loop(1) = {1};
Plane Surface(1) = {1};

// Mesh refinement
Field[1] = MathEval;
Field[1].F = "0.3*(Sqrt(x^2 + y^2))^0.8+1.0";
Background Field = 1;

// Options
Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.Algorithm = 6;