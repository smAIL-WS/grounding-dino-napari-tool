## Project Description

This project provides an **interactive, high-resolution annotation and inference tool** specifically tailored for agricultural computer vision. The system performs precision weed detection by leveraging a pretrained **Grounding DINO** model. The original implementation of this tool can be found here [UAV Weed Detection](https://github.com/smAIL-WS/uav_weed_detection.git).


### Key Capabilities

* **Precision Agriculture Focus:** Optimized for identifying "crop" and "weed" instances in complex, high-texture field environments.
* **Sliding Window Inference:** To maintain the spatial resolution necessary for detecting small weed sprouts in high-resolution drone or tractor-mounted imagery, the tool implements a sliding window protocol. This prevents the loss of detail caused by standard image resizing.
* **Interactive Refinement:** Integrated directly into the **Napari** multidimensional image viewer, the tool allows users to adjust confidence thresholds ($T_{conf}$) in real-time and manually correct or add bounding boxes to generate high-quality ground truth data.
* **Hybrid Architecture:** Supports both headless remote GPU server deployments (via FastAPI and Docker) and local workstation setups, ensuring high-performance inference regardless of the user's local hardware limitations.



This implementation guide covers two deployment strategies using **Docker**. Whether your GPU is on a remote server or right inside your laptop, using Docker ensures the complex AI dependencies (MMDetection, CUDA, PyTorch) are isolated and stable.



##  Scenario 1: Headless GPU Server (Remote Inference)
Use this if you have a powerful GPU server in a different location and you are using a standard laptop for the Napari interface.

### Phase 1: Server Setup (The "Engine")
1.  **Organize Files:** Create a folder named `dino_backend`. Place your `server.py`, `config.py`, and `Dockerfile` inside.
2.  **Prepare Models:** Create a subfolder `/checkpoints` and place your `.pth` files there.
3.  **Build Image:** Open the server terminal in that folder and run:
    `docker build -t dino-backend .`
4.  **Run Container:**
    ```bash
    docker run -d --gpus all \
      -p 8000:8000 \
      -v /home/user/dino_backend/checkpoints:/app/checkpoints \
      --name dino \
      dino-backend
    ```

### Phase 2: Local Laptop Setup (The "Interface")
1.  **Conda Env:** Open **Anaconda Prompt** on your laptop.
    ```bash
    conda create -n napari-dino python=3.10 -y
    conda activate napari-dino
    pip install "napari[all]" magicgui requests opencv-python-headless torch torchvision
    ```
2.  **SSH Tunnel (The Bridge):** Open a **new, standard Command Prompt (cmd)**.
    `ssh -L 8000:localhost:8000 your_user@your_server_ip`
    *Note: Keep this window open at all times while using the tool.*

### Phase 3: Launching
1.  In the **Anaconda Prompt**, navigate to your `plugin.py` folder.
2.  Run: `python plugin.py`

---

##  Scenario 2: Local GPU System (Dockerized)
Use this if the computer you are working on has an NVIDIA GPU. We still use Docker to avoid installing 50+ AI libraries directly on your Windows system.

### Phase 1: Local Docker Setup (The "Engine")
1.  **Organize Files:** Create a folder on your Windows machine (e.g., `C:\napari_tool`). Place `server.py`, `config.py`, and `Dockerfile` inside.
2.  **Prepare Models:** Create a folder `C:\napari_tool\checkpoints` and put your `.pth` files there.
3.  **Build Image:** Open **PowerShell** in that folder and run:
    `docker build -t dino-backend .`
4.  **Run Container:**
    ```powershell
    docker run -d --gpus all `
      -p 8000:8000 `
      -v C:\napari_tool\checkpoints:/app/checkpoints `
      --name dino `
      dino-backend
    ```
    *Note: You do NOT need an SSH tunnel for this setup.*

### Phase 2: Local Laptop Setup (The "Interface")
1.  **Conda Env:** Open **Anaconda Prompt**.
    ```bash
    conda create -n napari-dino python=3.10 -y
    conda activate napari-dino
    pip install "napari[all]" magicgui requests opencv-python-headless torch torchvision
    ```

### Phase 3: Launching
1.  Ensure the Docker container is running (`docker ps`).
2.  In the **Anaconda Prompt**, navigate to your `plugin.py` folder.
3.  Run: `python plugin.py`

---

##  Shared Troubleshooting & Diagnostics

### ### 1. Verify the Connection
Regardless of setup, open a web browser on your laptop and go to:
`http://localhost:8000/models`
* **Success:** You see a JSON list of your `.pth` files.
* **Failure:** You see "Site can't be reached."
    * *Remote Setup:* Your SSH tunnel is closed or port 8000 is blocked.
    * *Local Setup:* The Docker container crashed. Run `docker logs dino`.

### ### 2. Common Fixes
* **404 Error:** Your `server.py` is missing the `@app.get("/models")` route. Update the file and rebuild with `docker build --no-cache -t dino-backend .`
* **Predictions not appearing:** Ensure your `text_prompt` in the Napari widget matches the classes the model expects (e.g., `crop . weed`).
* **Color Logic:** The tool defaults **Blue** for labels containing "crop" and **Red** for labels containing "weed." Any other label will appear **White**.

### 3. Restarting the Engine
If you change your model weights or the `server.py` code, you must refresh the container:
```bash
docker stop dino
docker rm dino
# (Re-run the build and run commands from the specific scenario above)
```