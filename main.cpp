#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::dnn;

int main() {
   // 1. Load the ONNX model
   // JS Equivalent: const session = await ort.InferenceSession.create('./yolo12m.onnx');
   // Explanation: In JS we use ONNX Runtime Web. In C++, OpenCV has a built-in DNN module to load ONNX models.
   string modelPath = "./yolo/yolo12m.onnx"; // Update this path if your model is in a different folder

   cout << "Loading model from: " << modelPath << endl;
   Net net;
   try {
      net = readNetFromONNX(modelPath);
   } catch (const cv::Exception &e) {
      cerr << "Error loading model: " << e.what() << endl;
      cerr << "Make sure 'yolo12m.onnx' is in the same directory as the executable!" << endl;
      return -1;
   }

   // 2. Load the image
   // JS Equivalent: const img = document.getElementById('myImage');
   // Explanation: Reading an image file from disk into a matrix (array of pixels).
   string imagePath = "./images/img.jpg";
   Mat img = imread(imagePath);
   if (img.empty()) {
      cerr << "Error: Could not load image at " << imagePath << endl;
      return -1;
   }

   // 3. Preprocess the image (Resize, Normalize, Convert to Blob)
   // JS Equivalent:
   // const tensor = tf.browser.fromPixels(img)
   //                  .resizeBilinear([640, 640])
   //                  .div(255.0)
   //                  .expandDims(0);
   // Explanation: YOLO expects a 4D tensor [batch, channels, height, width], normalized between 0 and 1, and resized to 640x640.
   Mat blob;
   blobFromImage(img, blob, 1.0 / 255.0, Size(640, 640), Scalar(), true, false);

   // 4. Run Inference
   // JS Equivalent: const results = await session.run({ images: tensor });
   // Explanation: We feed the preprocessed blob into the network and get the output predictions.
   net.setInput(blob);
   vector<Mat> outputs;
   net.forward(outputs, net.getUnconnectedOutLayersNames());

   // 5. Post-processing (Parse YOLO output)
   // JS Equivalent: const data = results.output0.data; // Float32Array of shape [1, 84, 8400]
   // Explanation: YOLOv8/v12 outputs a tensor of shape [1, 84, 8400].
   // 84 = 4 bounding box coordinates (cx, cy, w, h) + 80 class probabilities.
   // 8400 = number of anchor boxes.
   Mat output = outputs[0];
   Mat out(output.size[1], output.size[2], CV_32F, output.ptr<float>());
   out = out.t(); // Transpose to [8400, 84] for easier iteration

   vector<int> classIds;
   vector<float> confidences;
   vector<Rect> boxes;

   float x_factor = img.cols / 640.0;
   float y_factor = img.rows / 640.0;

   // JS Equivalent: for (let i = 0; i < 8400; i++) { ... extract max confidence and box ... }
   for (int i = 0; i < out.rows; ++i) {
      Mat classesScores = out.row(i).colRange(4, 84);
      Point classIdPoint;
      double confidence;
      minMaxLoc(classesScores, 0, &confidence, 0, &classIdPoint);

      if (confidence > 0.5) { // Confidence threshold
         float cx = out.at<float>(i, 0);
         float cy = out.at<float>(i, 1);
         float w = out.at<float>(i, 2);
         float h = out.at<float>(i, 3);

         int left = int((cx - 0.5 * w) * x_factor);
         int top = int((cy - 0.5 * h) * y_factor);
         int width = int(w * x_factor);
         int height = int(h * y_factor);

         boxes.push_back(Rect(left, top, width, height));
         confidences.push_back((float)confidence);
         classIds.push_back(classIdPoint.x);
      }
   }

   // 6. Non-Maximum Suppression (NMS)
   // JS Equivalent: const indices = await tf.image.nonMaxSuppressionAsync(boxes, scores, maxOutputSize, iouThreshold, scoreThreshold);
   // Explanation: Removes overlapping bounding boxes for the same object.
   vector<int> indices;
   NMSBoxes(boxes, confidences, 0.5, 0.4, indices);

   // 7. Draw Bounding Boxes
   // JS Equivalent: ctx.strokeRect(x, y, w, h); ctx.fillText(label, x, y);
   // Explanation: Loop through the kept boxes and draw them on the original image.
   for (int idx : indices) {
      Rect box = boxes[idx];
      rectangle(img, box, Scalar(0, 255, 0), 2);

      string label = format("Class %d: %.2f", classIds[idx], confidences[idx]);
      putText(img, label, Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
   }

   // 8. Show the result
   // JS Equivalent: ctx.putImageData(imgData, 0, 0);
   // Explanation: Display the image in a window.
   imshow("YOLO Object Detection", img);
   waitKey(0);

   return 0;
}
