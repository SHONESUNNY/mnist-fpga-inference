
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np
from pathlib import Path

# ---------------- Configuration ----------------
CANVAS_SIZE = 280
MODEL_SIZE = 28
DIGIT_SIZE = 20
OUTPUT_DIR = Path("digit_output")

OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------- Application ----------------
class DigitDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title(" Digit Recognition ")

        # Drawing surface: black background, white digit.
        self.canvas = tk.Canvas(
            root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            cursor="cross",
            highlightthickness=1
        )
        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        self.image = Image.new(
            "L", (CANVAS_SIZE, CANVAS_SIZE), 0
        )
        self.draw = ImageDraw.Draw(self.image)

        self.last_x = None
        self.last_y = None

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        tk.Button(
            root, text="Clear", command=self.clear
        ).grid(row=1, column=0, padx=5, pady=5)

        tk.Button(
            root, text="Preprocess", command=self.preprocess
        ).grid(row=1, column=1, padx=5, pady=5)

        tk.Button(
            root, text="Save 784 Bytes", command=self.save
        ).grid(row=1, column=2, padx=5, pady=5)

        self.preview_label = tk.Label(root, text="28 x 28 preview")
        self.preview_label.grid(row=2, column=0, columnspan=3)

        self.info = tk.Label(root, text="Draw a digit from 0 to 9.")
        self.info.grid(row=3, column=0, columnspan=3, pady=8)

        self.pixel_array = None
        self.preview_photo = None

    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y

        # A dot should appear even for a short click.
        r = 6
        self.canvas.create_oval(
            event.x-r, event.y-r,
            event.x+r, event.y+r,
            fill="white", outline="white"
        )
        self.draw.ellipse(
            (event.x-r, event.y-r, event.x+r, event.y+r),
            fill=255
        )

    def paint(self, event):
        if self.last_x is None:
            return

        # Draw on both the GUI and the actual PIL image.
        self.canvas.create_line(
            self.last_x, self.last_y, event.x, event.y,
            fill="white", width=12,
            capstyle=tk.ROUND, smooth=True
        )

        self.draw.line(
            (self.last_x, self.last_y, event.x, event.y),
            fill=255, width=12
        )

        r = 6
        self.draw.ellipse(
            (event.x-r, event.y-r, event.x+r, event.y+r),
            fill=255
        )

        self.last_x = event.x
        self.last_y = event.y

    def stop_draw(self, event):
        self.last_x = None
        self.last_y = None

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new(
            "L", (CANVAS_SIZE, CANVAS_SIZE), 0
        )
        self.draw = ImageDraw.Draw(self.image)
        self.pixel_array = None
        self.preview_label.config(image="", text="28 x 28 preview")
        self.preview_photo = None
        self.info.config(text="Canvas cleared.")

    def preprocess(self):
        # Convert to grayscale and locate the digit.
        gray = self.image.convert("L")
        arr = np.array(gray)

        # Ignore very faint pixels near the background.
        ys, xs = np.where(arr > 20)

        if len(xs) == 0:
            messagebox.showwarning(
                "Empty canvas", "Draw a digit first."
            )
            return

        # Crop to the digit's bounding box.
        cropped = gray.crop((
            int(xs.min()), int(ys.min()),
            int(xs.max()) + 1, int(ys.max()) + 1
        ))

        # Preserve aspect ratio by padding to a square.
        w, h = cropped.size
        side = max(w, h)
        square = Image.new("L", (side, side), 0)
        square.paste(
            cropped, ((side - w) // 2, (side - h) // 2)
        )

        # Fit digit within 20 x 20, leaving a 4-pixel border
        # around the 28 x 28 image.
        fitted = square.resize(
            (DIGIT_SIZE, DIGIT_SIZE),
            Image.Resampling.LANCZOS
        )

        # Place in the center of a black 28 x 28 image.
        result = Image.new("L", (MODEL_SIZE, MODEL_SIZE), 0)
        result.paste(fitted, (4, 4))

        # Store exactly 784 row-major unsigned pixel values.
        self.pixel_array = np.asarray(
            result, dtype=np.uint8
        ).flatten()

        # Show an enlarged preview using nearest-neighbor scaling.
        enlarged = result.resize(
            (MODEL_SIZE * 8, MODEL_SIZE * 8),
            Image.Resampling.NEAREST
        )
        self.preview_photo = ImageTk.PhotoImage(enlarged)
        self.preview_label.config(
            image=self.preview_photo, text=""
        )

        self.info.config(
            text=(
                f"Shape: 28 x 28 | Bytes: {self.pixel_array.size} | "
                f"Range: {self.pixel_array.min()}–"
                f"{self.pixel_array.max()}"
            )
        )

        # Save the exact 28 x 28 image for inspection.
        result.save(OUTPUT_DIR / "sent_image.png")

    def save(self):
        if self.pixel_array is None:
            self.preprocess()

        if self.pixel_array is None:
            return

        # Save the exact raw bytes intended for FPGA transmission.
        payload = self.pixel_array.tobytes()
        binary_path = OUTPUT_DIR / "sent_image.bin"
        binary_path.write_bytes(payload)

        # Also save a NumPy representation for debugging.
        np.save(
            OUTPUT_DIR / "sent_image.npy",
            self.pixel_array.reshape(28, 28)
        )

        # Verify the file size and pixel count.
        assert len(payload) == 784
        assert len(payload) == 28 * 28

        messagebox.showinfo(
            "Saved",
            f"Saved {len(payload)} bytes.\n\n"
            f"Image: {OUTPUT_DIR / 'sent_image.png'}\n"
            f"Binary: {binary_path}\n"
            f"Array: {OUTPUT_DIR / 'sent_image.npy'}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitDrawingApp(root)
    root.mainloop()
