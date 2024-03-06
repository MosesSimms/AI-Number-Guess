import raylib
from pyray import *
from PIL import Image
import numpy as np
from numpy import asarray
import torchvision.transforms.functional
from predict_num import *
from torch import Tensor

screenWidth = 256
screenHeight = 255

init_window(screenWidth, screenHeight, "AI Number Guesser")

set_target_fps(60)

target = load_render_texture(screenWidth, screenHeight)
brushSize = 10

begin_texture_mode(target)
clear_background(BLACK)
end_texture_mode()

while not window_should_close():

    #drawing = load_image_from_screen()
    

    mousePosition = get_mouse_position()

    if is_key_pressed(raylib.KEY_C):
        begin_texture_mode(target)
        clear_background(BLACK)
        end_texture_mode()

    if is_mouse_button_down(raylib.MOUSE_BUTTON_LEFT):

        begin_texture_mode(target)
        draw_circle(int(mousePosition.x), int(mousePosition.y), float(brushSize), WHITE)
        end_texture_mode()

    if is_key_pressed(raylib.KEY_ENTER):
        drawing2 = load_image_from_texture(target.texture)
        image_flip_vertical(drawing2)
        export_image(drawing2, "drawing.png")

        drawing = Image.open("./drawing.png")
        drawing = drawing.convert("L")
        newsize = (28,28)
        drawing = drawing.resize(newsize)
    


        drawing.save("drawing.png")
        numpydata = asarray(drawing)
        print(numpydata.shape)

        drawing_data = torchvision.transforms.functional.to_tensor(np.array(numpydata).T).to(device)
        drawing_data = torchvision.transforms.functional.vflip(drawing_data)
        drawing_data = drawing_data.permute(0, 2, 1)
        drawing_data = np.squeeze(drawing_data)
        
        output = network(drawing_data)
        output = Tensor.detach(output).to("cpu").numpy()
        prediction = np.argmax(output)


        print("\nComputer's Guess: ", prediction)


    begin_drawing()

    clear_background(BLACK)

    draw_texture_rec(target.texture, Rectangle(0, 0, float(screenWidth), float(screenHeight * -1)), (0, 0), RED)

    draw_circle(int(mousePosition.x), int(mousePosition.y), float(brushSize), WHITE)

    end_drawing()

    
unload_render_texture(target)
close_window()



