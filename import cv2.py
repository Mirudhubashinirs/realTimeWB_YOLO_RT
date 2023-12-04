import cv2
import inference
import supervision as sv

annotator = sv.BoxAnnotator()

def on_prediction(predictions, image):
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_roboflow(predictions)
    detections = detections[detections.confidence > 0.9]
    print(detections)
    cv2.imshow(
        "Prediction", 
        annotator.annotate(
            scene=image, 
            detections=detections,
            labels=labels
        )
    )
    cv2.waitKey(1)
      # Corrected function call

# Use your own API key and model ID
api_key = 'qNemsO9IKeCPkhBFUs0D'
model_id = 'fd-ep-new/1'

inference.Stream(
    source= 1,
    api_key=api_key,
    model=model_id,
    output_channel_order="BGR",
    use_main_thread=True,
    on_prediction=on_prediction, 
)
