# Built in packages
import math
import sys
from pathlib import Path
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import imageIO.png

# Function that reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # Pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# A useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array

#Function that takes the 3 seperate arrays and combines them to form a RGB array
def seperateArraysToRGB( px_array_r, px_array_g, px_array_b, image_width, image_height):
    new_array = [[[0 for c in range(3)] for x in range(image_width)] for y in range(image_height)]

    for y in range(image_height):
        for x in range(image_width):
            new_array[y][x][0] = px_array_r[y][x]
            new_array[y][x][1] = px_array_g[y][x]
            new_array[y][x][2] = px_array_b[y][x]

    return new_array

#Function that takes a RGB image and converts it to GreyScale
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height, 0)
    
    for y in range(image_height):
        for x in range(image_width):
            greyscale_pixel_array[y][x] = round((0.299 * pixel_array_r[y][x]) + (0.587 * pixel_array_g[y][x]) + (0.114 * pixel_array_b[y][x]))
    
    return greyscale_pixel_array

#Function that Scales each pixel to 0 and 255
def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    
    image = createInitializedGreyscalePixelArray(image_width, image_height, 0)
    
    sout = 0
    flow = 255
    fhigh = 0
    
    for x in range(image_height):
        for y in range(image_width):
            #Getting lowest and highest value in pixel array
            if pixel_array[x][y] < flow:
                flow = pixel_array[x][y]
            elif pixel_array[x][y] > fhigh:
                fhigh = pixel_array[x][y]
                
    #Difference between biggest and smallest value in pixel array
    bottom = fhigh - flow
    
    for i in range(image_height):
        for j in range(image_width):
            if bottom == 0:
                image[i][j] = 0
                continue
            else:
                sout = round((pixel_array[i][j] - flow) * (255 / bottom))
            
            if sout < 0:
                image[i][j] = 0
            elif sout > 255:
                image[i][j] = 255
            else:
                image[i][j] = sout
    
    return image

#Helper Function that calculates the mean of a 5x5 kernal around the pixel
def meanOf5x5(pixel_array, image_width, image_height, i, j):
    
    total = 0

    for x in range(-2, 3):
        for y in range(-2, 3):
            total += pixel_array[i + x][j + y] 

    mean = total / 25

    return mean

#Helper Function that calculates the variance of a 5x5 kernal around the pixel
def varianceOf5x5(pixel_array, image_width, image_height, i, j, mean):

    total = 0

    for x in range(-2, 3):
        for y in range(-2, 3):
            total += pow(pixel_array[i + x][j + y] - mean, 2)

    variance = total / 25

    return variance

#Function that computes the standard deviation
def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):

    image = createInitializedGreyscalePixelArray(image_width, image_height, 0.0)

    for i in range (2, image_height-2):
        for j in range (2, image_width-2):
            mean = meanOf5x5(pixel_array, image_width, image_height, i, j)

            variance = varianceOf5x5(pixel_array, image_width, image_height, i, j, mean)

            standardDeviation = math.sqrt(variance)

            image[i][j] = standardDeviation

    return image

#Helper Function that creates a Padding around the image
def RepeatedBorderBoundaryPadding(new, old, image_width, image_height):
    padded_width = image_width + 2
    padded_height = image_height + 2
    
    for row in range(image_height):
        for col in range(image_width):
            new[row + 1][col + 1] = old[row][col]
    
    top_row = old[0]
    bottom_row = old[image_height - 1]
    left_column = [old[i][0] for i in range(image_height)]
    right_column = [old[i][image_width - 1] for i in range(image_height)]
    
    new[0] = [old[0][0]] + top_row + [old[0][image_width - 1]]
    new[padded_height - 1] = [old[image_height - 1][0]] + bottom_row + [old[image_height - 1][image_width - 1]]
    
    for i in range(image_height):
        new[i + 1][0] = left_column[i]
        new[i + 1][padded_width - 1] = right_column[i]
    
    return new

#Helper Function that calculates the Gaussian Value   
def GaussianValue(MWT):
    Gauss = [[1/16, 1/8, 1/16], 
            [1/8,1/4,1/8], 
            [1/16, 1/8, 1/16]]
    
    count = 0
    for i in range(3):
        for j in range(3):
            count = count + MWT[i][j] * Gauss[i][j]
    
    return (count)


#Function that does the Gaussian Filter
def computeGaussianAveraging3x3RepeatBorder(pixel_array, image_width, image_height):
    
    k = 1
    l = 1
    
    image = createInitializedGreyscalePixelArray(image_width, image_height)
    temp = createInitializedGreyscalePixelArray(image_width + 2*k, image_height + 2*l)
    #Array with repeated borders
    padded = (RepeatedBorderBoundaryPadding(temp, pixel_array, image_width, image_height))
    
    for row in range(1, image_height+1):
        for col in range(1, image_width+1):
            Kernal3x3 = []
            
            for i in range(row - 1, row + 2):
                tempList = []
                for j in range(col -1, col+2):
                    tempList.append(padded[i][j])
                
                Kernal3x3.append(tempList)

            image[row - 1][col - 1] = (GaussianValue(Kernal3x3))
    
    return image


def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # This is the default input image filename
    filename = "Barcode1"
    input_filename = "images/"+filename+".png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(filename+"_output.png")
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    # We read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    #Takes the original seperate arrays and combines it into an RGB array
    px_array = seperateArraysToRGB(px_array_r, px_array_g, px_array_b, image_width, image_height)

    #Turns the RGB Image to Greyscale
    grey_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height) 

    #Scales the Image to pixel values of 0 and 255
    normal_px_array = scaleTo0And255AndQuantize(grey_array, image_width, image_height)  

    #Applies a 5x5 Standard Deviation Filter
    standard_px_array = computeStandardDeviationImage5x5(normal_px_array, image_width, image_height)

    #Applies a 3x3 Guassian Filter 4 Times
    guassian1_px_array = computeGaussianAveraging3x3RepeatBorder(standard_px_array, image_width, image_height)
    guassian2_px_array = computeGaussianAveraging3x3RepeatBorder(guassian1_px_array, image_width, image_height)
    guassian3_px_array = computeGaussianAveraging3x3RepeatBorder(guassian2_px_array, image_width, image_height)
    guassian4_px_array = computeGaussianAveraging3x3RepeatBorder(guassian3_px_array, image_width, image_height)
    
    # Compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
    # Change these values based on the detected barcode region from your algorithm
    center_x = image_width / 2.0
    center_y = image_height / 2.0
    bbox_min_x = center_x - image_width / 4.0
    bbox_max_x = center_x + image_width / 4.0
    bbox_min_y = center_y - image_height / 4.0
    bbox_max_y = center_y + image_height / 4.0

    # The following code is used to plot the bounding box and generate an output for marking
    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # Write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # Plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()