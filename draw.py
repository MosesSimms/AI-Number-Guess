import raylib
from pyray import *
from PIL import Image
import numpy as np
from numpy import asarray
import torchvision.transforms.functional
from predict_num import *
from torch import Tensor

screenWidth = 256
screenHeight = 256

init_window(screenWidth, screenHeight, "AI Number Guesser")
set_target_fps(60)

target = load_render_texture(screenWidth, screenHeight)
brushSize = 10

# Clearing Texture
begin_texture_mode(target)
clear_background(BLACK)
end_texture_mode()

while not window_should_close():

    mousePosition = get_mouse_position()

    # Clear Drawing with 'C' 
    if is_key_pressed(raylib.KEY_C):
        begin_texture_mode(target)
        clear_background(BLACK)
        end_texture_mode()

    # Starts Drawing When Left Mouse Button Is Held Down
    if is_mouse_button_down(raylib.MOUSE_BUTTON_LEFT):

        begin_texture_mode(target)
        draw_circle(int(mousePosition.x), int(mousePosition.y), float(brushSize), WHITE)
        end_texture_mode()

    # Generates Image And Computer Guess on 'ENTER'
    if is_key_pressed(raylib.KEY_ENTER):

        # Saves Texture As Image Because Pillow Cant Read C Struct
        drawing = load_image_from_texture(target.texture)
        # Flipping The Image To Match Format Of Training Data
        image_flip_vertical(drawing)
        export_image(drawing, "drawing.png")

        # Converting Image To Greyscale And Resizing To Match Training Data Format
        drawing = Image.open("./drawing.png")
        drawing = drawing.convert("L")
        newsize = (28,28)
        drawing = drawing.resize(newsize)

        # Saves Formatted Image for User Reference
        drawing.save("drawing.png")

        # Converting Image To Numpy Array
        numpydata = asarray(drawing)

        # Converting Array To Tensor And Formatting For Network
        drawing_data = torchvision.transforms.functional.to_tensor(np.array(numpydata).T).to(device)
        drawing_data = torchvision.transforms.functional.vflip(drawing_data)
        drawing_data = drawing_data.permute(0, 2, 1)
        drawing_data = np.squeeze(drawing_data)
        
        # Run Network Against Tensor
        output = network(drawing_data)
        output = Tensor.detach(output).to("cpu").numpy()
        prediction = np.argmax(output)


        print("\nComputer's Guess: ", prediction)


    begin_drawing()

    clear_background(BLACK)

    # Draws Texture To Cover The Whole Window
    draw_texture_rec(target.texture, Rectangle(0, 0, float(screenWidth), float(screenHeight * -1)), (0, 0), RED)

    # Tracks Mouse Position For User Reference
    draw_circle(int(mousePosition.x), int(mousePosition.y), float(brushSize), WHITE)

    end_drawing()

    
unload_render_texture(target)
close_window()



