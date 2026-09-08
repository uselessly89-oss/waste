# Easy Camera Mode

This mode is intended for the easiest demonstration of the waste model.

## Laptop camera

```bash
pip install -r requirements-app.txt
python app.py
```

Open the local Gradio address in the browser and select **Webcam**.

## Phone camera

Run `python app.py` on a laptop connected to the same Wi-Fi network as the phone.
Find the laptop's LAN IP (for example `192.168.1.20`) and open:

`http://<LAPTOP-IP>:7860`

in the phone browser. Choose the camera input and point the phone at the waste.

If the network blocks port 7860, use the normal laptop camera or configure the
network/firewall accordingly.

## Mixed clusters

Cluster mode reports every confidently visible object. The model does **not**
pretend that hidden objects can be identified from RGB pixels. Low-confidence,
unknown, or visually inseparable material must be routed to `REJECT` in an
actual sorting system. For a whole mixed pile, the correct production design
is segmentation plus depth/weight/NIR or another suitable sensor and a second
pass after items are separated.

## Important

The app needs trained weights. Set `WASTE_MODEL=/path/to/best.pt` if the weights
are not at `runs/waste/waste_detector/weights/best.pt`.

This camera app is for perception/demo. Keep physical actuators disabled until
the detector, tracking, timing, emergency stop, and mechanical system have
been validated independently.
