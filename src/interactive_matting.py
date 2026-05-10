#!/usr/bin/env python3
"""Interactive Image Matting with real-time preview"""

import cv2
import numpy as np
import argparse
import os
import sys


class InteractiveMatting:
    """Interactive scribble-based matting tool"""

    def __init__(self, image_path):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from: {image_path}")

        self.image_path = image_path
        self.height, self.width = self.original_image.shape[:2]

        self.display_image = self.original_image.copy()
        self.scribble_mask = np.full((self.height, self.width), 128, dtype=np.uint8)

        self.alpha_matte = None
        self.composite = None

        self.drawing = False
        self.brush_size = 5
        self.current_mode = None
        self.prev_point = None

        self.fg_color = (0, 0, 255)
        self.bg_color = (255, 0, 0)

        self.scribble_window = "1. Draw Scribbles (Space=Process, C=Clear, Q=Quit)"
        self.alpha_window = "2. Alpha Matte Result"
        self.composite_window = "3. Composite (Press B to change background)"

        self.backgrounds = ['white', 'black', 'green', 'red', 'blue', 'checkerboard']
        self.current_bg_idx = 2  # Start with green (index 2)

        sys.path.insert(0, os.path.dirname(__file__))
        from closed_form_matting import solve_closed_form_matting
        self.solve_matting = solve_closed_form_matting

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_mode = 'foreground'
            self.prev_point = (x, y)
            self.draw_scribble(x, y)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.current_mode = 'background'
            self.prev_point = (x, y)
            self.draw_scribble(x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.draw_scribble(x, y)
                self.prev_point = (x, y)

        elif event in [cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP]:
            self.drawing = False
            self.current_mode = None
            self.prev_point = None

    def draw_scribble(self, x, y):
        if self.current_mode == 'foreground':
            cv2.circle(self.display_image, (x, y), self.brush_size, self.fg_color, -1)
            cv2.circle(self.scribble_mask, (x, y), self.brush_size, 255, -1)
            if self.prev_point is not None:
                cv2.line(self.display_image, self.prev_point, (x, y),
                        self.fg_color, self.brush_size * 2)
                cv2.line(self.scribble_mask, self.prev_point, (x, y),
                        255, self.brush_size * 2)

        elif self.current_mode == 'background':
            cv2.circle(self.display_image, (x, y), self.brush_size, self.bg_color, -1)
            cv2.circle(self.scribble_mask, (x, y), self.brush_size, 0, -1)
            if self.prev_point is not None:
                cv2.line(self.display_image, self.prev_point, (x, y),
                        self.bg_color, self.brush_size * 2)
                cv2.line(self.scribble_mask, self.prev_point, (x, y),
                        0, self.brush_size * 2)
                
    def create_checkerboard(self, square_size=20):
        """Create a checkerboard pattern background."""
        h, w = self.height, self.width
        checkerboard = np.zeros((h, w, 3), dtype=np.uint8)

        for i in range(0, h, square_size):
            for j in range(0, w, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    checkerboard[i:i+square_size, j:j+square_size] = [200, 200, 200]
                else:
                    checkerboard[i:i+square_size, j:j+square_size] = [100, 100, 100]

        return checkerboard
    
    def get_background(self):
        """Get the current background based on selected index."""
        bg_name = self.backgrounds[self.current_bg_idx]

        if bg_name == 'white':
            return np.ones_like(self.original_image) * [255, 255, 255]
        elif bg_name == 'black':
            return np.zeros_like(self.original_image)
        elif bg_name == 'green':
            return np.ones_like(self.original_image) * [0, 255, 0]
        elif bg_name == 'red':
            return np.ones_like(self.original_image) * [0, 0, 255]  # BGR format
        elif bg_name == 'blue':
            return np.ones_like(self.original_image) * [255, 0, 0]  # BGR format
        elif bg_name == 'checkerboard':
            return self.create_checkerboard()
        else:
            return np.ones_like(self.original_image) * [0, 255, 0]  # Default green

    def clear_scribbles(self):
        self.display_image = self.original_image.copy()
        self.scribble_mask = np.full((self.height, self.width), 128, dtype=np.uint8)
        self.alpha_matte = None
        self.composite = None
        print("cleared all scribbles")

    def process_matting(self):
        fg_pixels = np.sum(self.scribble_mask == 255)
        bg_pixels = np.sum(self.scribble_mask == 0)

        if fg_pixels == 0 or bg_pixels == 0:
            print(f"need both fg and bg scribbles! current: fg={fg_pixels}, bg={bg_pixels}")
            return

        print(f"processing matting... fg={fg_pixels}, bg={bg_pixels}, unknown={np.sum(self.scribble_mask == 128)}")

        try:
            image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB) / 255.0

            print("running matting algorithm...")
            alpha = self.solve_matting(image_rgb, self.scribble_mask, lambda_param=100)
            self.alpha_matte = (alpha * 255).astype(np.uint8)

            # Use the background selection function
            bg = self.get_background()
            composite = self.original_image * alpha[:, :, np.newaxis] + \
                       bg * (1 - alpha[:, :, np.newaxis])
            self.composite = composite.astype(np.uint8)

            print(f"matting done! alpha range: [{alpha.min():.3f}, {alpha.max():.3f}]")
            print("results in windows 2 and 3, press space to refine")

        except Exception as e:
            print(f"error during matting: {e}")
            import traceback
            traceback.print_exc()

    def save_results(self):
        if self.alpha_matte is None:
            print("no results to save - press space to process first!")
            return

        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        output_dir = 'output/interactive'
        os.makedirs(output_dir, exist_ok=True)

        scribbles_path = os.path.join(output_dir, f"{base_name}_scribbles.png")
        cv2.imwrite(scribbles_path, self.scribble_mask)

        viz_path = os.path.join(output_dir, f"{base_name}_scribbles_viz.png")
        cv2.imwrite(viz_path, self.display_image)

        alpha_path = os.path.join(output_dir, f"{base_name}_alpha.png")
        cv2.imwrite(alpha_path, self.alpha_matte)

        composite_path = os.path.join(output_dir, f"{base_name}_composite.png")
        cv2.imwrite(composite_path, self.composite)

        print(f"saved to: {scribbles_path}, {alpha_path}, {composite_path}")

    def draw_instructions(self, image):
        img = image.copy()

        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (500, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 35
        cv2.putText(img, "LEFT CLICK: Draw foreground (RED)", (20, y),
                   font, 0.5, (0, 0, 255), 1)
        y += 25
        cv2.putText(img, "RIGHT CLICK: Draw background (BLUE)", (20, y),
                   font, 0.5, (255, 0, 0), 1)
        y += 25
        cv2.putText(img, "SPACE: Process matting", (20, y),
                   font, 0.5, (0, 255, 0), 1)
        y += 25
        cv2.putText(img, "+/- : Change brush size", (20, y),
                   font, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(img, "C: Clear scribbles  |  S: Save results", (20, y),
                   font, 0.5, (255, 255, 255), 1)   
        y += 25
        cv2.putText(img, "B: Change background", (20, y),
                font, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(img, "Q/ESC: Quit", (20, y),
                   font, 0.5, (255, 255, 255), 1)

        return img

    def run(self):
        print("interactive image matting")
        print("instructions:")
        print("  1. draw red scribbles on foreground (left click)")
        print("  2. draw blue scribbles on background (right click)")
        print("  3. press space to process and see results")
        print("  4. refine by adding more scribbles and pressing space again")
        print("  5. press s to save, q to quit")

        # Create windows
        cv2.namedWindow(self.scribble_window)
        cv2.setMouseCallback(self.scribble_window, self.mouse_callback)

        while True:
            display = self.draw_instructions(self.display_image)
            cv2.imshow(self.scribble_window, display)

            if self.alpha_matte is not None:
                alpha_colored = cv2.applyColorMap(self.alpha_matte, cv2.COLORMAP_JET)
                cv2.imshow(self.alpha_window, alpha_colored)
                cv2.imshow(self.composite_window, self.composite)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                print("quitting...")
                break

            elif key == ord(' '):
                self.process_matting()

            elif key == ord('c'):
                self.clear_scribbles()
                cv2.destroyWindow(self.alpha_window)
                cv2.destroyWindow(self.composite_window)

            elif key == ord('s'):
                self.save_results()

            elif key == ord('b'):
                if self.alpha_matte is not None:
                    # Cycle to next background
                    self.current_bg_idx = (self.current_bg_idx + 1) % len(self.backgrounds)

                    # Regenerate composite with new background
                    bg = self.get_background()
                    alpha = self.alpha_matte / 255.0
                    self.composite = (self.original_image * alpha[:, :, np.newaxis] +
                                    bg * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)

                    print(f"background: {self.backgrounds[self.current_bg_idx]}")
                else:
                    print("process matting first (press space) before changing background")

            elif key == ord('+') or key == ord('='):
                self.brush_size = min(20, self.brush_size + 1)
                print(f"brush size: {self.brush_size}")

            elif key == ord('-') or key == ord('_'):
                self.brush_size = max(1, self.brush_size - 1)
                print(f"brush size: {self.brush_size}")

        cv2.destroyAllWindows()
        print("done!")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive image matting with scribble refinement"
    )
    parser.add_argument(
        "image_path",
        help="Path to input image"
    )
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        sys.exit(1)

    tool = InteractiveMatting(args.image_path)
    tool.run()


if __name__ == "__main__":
    main()
