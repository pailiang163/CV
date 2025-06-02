from ultralytics import YOLOWorld
from ultralytics import YOLO

# Initialize a YOLO-World model
# model = YOLOWorld("yolov8x-worldv2.pt")

# # Define custom classes
# model.set_classes(["Advertising stickers","Square sticker advertising","Green square sign","Green special equipment identification plate"])  #"A person lying on the ground"

# # Execute prediction on an image
# results = model.predict(r"images\2e0662eae9d27ce35e158dd17fe8f5f.jpg")

# # Show results
# results[0].show()

model = YOLO("yoloe-11l-seg.pt")
names = ["Square sticker advertising","Electronic screen","Square sticker with green frame","Border green background yellow square sticker",
          "White stickers","Yellow square sign with green border background"]
model.set_classes(names, model.get_text_pe(names))
results = model.predict(r"images\2e0662eae9d27ce35e158dd17fe8f5f.jpg")
results[0].show()
results[0].save("yoloe.jpg")