## Simulator
Run the My Project.exe file to start the simulation

![Alt text](../Assets/Man_Unity.png)

Then run the CameraFeed.py to the camera feeds, that you will get through tcp stream.

![Alt text](../Assets/CameraFeeds.png)

Use this stream to do Augmented reality on it.

## Movement
* Use tcp port 5005 for displaying, moving and rotating any image to project
* Use tcp port 5006 for moving, rotating the fbx file

## How to Use:
 * Run the My Project.exe to start the simulator,
 * and run the CameraFeed.py script to view the camera feed,( if u dont have the dependencies, run "./.venv/bin/activate" in the terminal, and run the python file, this virtual environment should have all the dependencies installed.)
* Use move_img.py to move the image and the fbx file.
