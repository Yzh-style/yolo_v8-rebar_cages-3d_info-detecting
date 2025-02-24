from ultralytics import YOLO
import multiprocessing

def train_model(epochs, imgsz, workers, batch_size, pretrained, device=0, 
                mosaic=False, box=10.5, cls=5.5,  patience=50,verbose= True, 
                lr0=0.01, lrf=0.01, optimizer='auto', overlap_mask=True, 
                val=True, plots=True,  degrees=5, flipud=0.2, scale=0.1):#
    # Load a model
    model = YOLO(r"D:\Desktop\ultralytics\ultralytics\runs\segment\train4\weights\best.pt")
    # Train the model
    model.train(data=r"D:\Desktop\ultralytics\ultralytics\cfg\datasets\coco128-seg.yaml",
                epochs=epochs,
                imgsz=imgsz,
                workers=workers,
                batch=batch_size,
                pretrained=pretrained,                        
                patience=patience,
                device=device,
                mosaic=False,
                lr0=lr0,
                lrf=lrf,
                optimizer=optimizer,
                overlap_mask=overlap_mask,
                val=val,
                degrees=degrees,
                flipud=flipud,
                scale=scale,
         
                # Add mosaic parameter to augmentations
                )
     
if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Define hyperparameters grid
    imgsz_list = [760]
    pretrained_list = [True]
    optimizer_list = ['auto']  # Add optimizer options here
    
    # Grid search
    for imgsz in imgsz_list:
        for pretrained in pretrained_list:
            for optimizer in optimizer_list:
                train_model(epochs=300, imgsz=imgsz, workers=1, batch_size=2, 
                            pretrained=pretrained, mosaic=False, device=0,  
                            patience=25, optimizer=optimizer, overlap_mask=True, 
                            val=True)
                print(f"Training model with imgsz={imgsz}, pretrained={pretrained}, optimizer={optimizer}")

