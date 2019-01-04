import cv2
import numpy as np

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
color = (255, 255, 255)
thickness = -1
im = None

# mouse callback function
def draw_mask(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode, color, thickness
    global mask, im, original

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                pt1 = (current_former_x+10,current_former_y+10)
                pt2 = (former_x-10,former_y-10)
                cv2.rectangle(mask, pt1, pt2, color, thickness)
                current_former_x = former_x
                current_former_y = former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            pt1 = (current_former_x+10,current_former_y+10)
            pt2 = (former_x-10,former_y-10)
            cv2.rectangle(mask, pt1, pt2, color, thickness)
            current_former_x = former_x
            current_former_y = former_y
    im = cv2.bitwise_not(cv2.bitwise_not(mask) * original)
    # im = original
    return former_x,former_y

def run_draw(img, output_name="mask.png"):
    global im, mask, original
    im = img
    original = im.copy()
    mask = np.zeros_like(im)

    cv2.namedWindow("Create a mask")
    cv2.setMouseCallback('Create a mask', draw_mask)
    while(1):
        cv2.imshow('Create a mask',im)
        k=cv2.waitKey(1)&0xFF
        if k==27:
            break
    cv2.destroyAllWindows()

    print("Save image")
    cv2.imwrite(output_name, mask)

if __name__ == "__main__":
    img = cv2.imread("images/bridge.jpg")
    run_draw(img)
