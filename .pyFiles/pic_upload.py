# def picture(image_path, transform, model, face_detect=False):
#   if face_detect:
#     detector = face_detection.build_detector(
#           "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
#     image = cv2.imread(image_path)
#     im = Image.open(image_path)
#     im2 = transform(im).permute(1, 2, 0)
#     pic = plt.imshow(im)
#     dets = detector.detect(
#                         image[:, :, ::-1]
#                     )[:, :4]
#     for bbox in dets:
#         top, right, bottom, left = [int(_) for _ in bbox]
#         cv2.rectangle(image, (top, right), (bottom, left), (0, 0, 255), 2)
#         img = TF.to_tensor(im)[top:bottom, left:right].permute(2, 0, 1).unsqueeze(0)
#         img = test_transform(img)
#         age = int(model(img).item())
#         label = f'age:{age}'
#         (w, h), _ = cv2.getTextSize(
#             label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
#         cv2.putText(image, label, (bottom, right - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


#   else:
#     img = Image.open(image_path).convert('RGB')

#     img_tensor = transform(img).unsqueeze(0)
#   with torch.inference_mode():
#     preds = model(img_tensor.to(device)).item()

#   return preds, img