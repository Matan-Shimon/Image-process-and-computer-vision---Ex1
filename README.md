**Full name** - Matan Yarin Shimon

**ID** - 314669342

**Python version** - 3.9

**Platform** - PyCharm
### Submitted files
1. **ex1_utils.py** - This file contains all of the functions that iv'e written for this project (except the gammaDisplay function).
2. **gamma.py** - This file contains the gammaDisplay function.
3. **ex1_main.py** - This file is the 'test file', using this file i chcecked if the functions that iv'e written in ex1_utils.py are well performed, this file has been provided to me.
4. **testImg1.jpg** - This is the first image testing file (this image contains beautiful sunset).
5. **testImg2.jpg** - This is the second image testing file (this image contains the breathtaking Eiffel Tower).
6. **beach.jpg** - an image I have received with the project.
7. **dark.jpg** - an image I have received with the project.
8. **bac_con.png** - an image I have received with the project.
9. **water_bear.png** - an image I have received with the project.

### Functions
#### for more detailed explanation go to ex1_utils.py where there I have written full documentation of each function proccess
1. **myID** - This function returns my ID.
2. **imReadAndConvert** - This function gets a filename string and an integer represantation, the function returns the image as a multidimensional numpy array based on the representation the function received.
3. **imDisplay** - This function gets filename string and an integer representation, the function desplays the image based on the desired representation.

![gray_paris](https://user-images.githubusercontent.com/63747865/161266472-2a309af2-b935-42c1-9b5a-4ca9c2b1b1ec.png)
![rgb_paris](https://user-images.githubusercontent.com/63747865/161266493-ba6640fb-2585-4964-bad3-5a1412f10d07.png)

4. **transformRGB2YIQ** - This function recieves an image as RGB type, the function returns the image converted to YIQ type.
5. **transformYIQ2RGB** - This function recieves an image as YIQ type, the function returns the image converted to RGB type.
![image](https://user-images.githubusercontent.com/63747865/161268046-23e6e36a-c83c-47e5-91a2-fab6cd387703.png)

6. **hsitogramEqualize** - This function gets an image, the function returns the image with maximum constrast, the original historam and the maximum constrast image histogram.

Proccess: if the image is RGB type, then we first convert it to YIQ and performing the changes only on the Y channel. normalizing the image, calculting the original image histogram and cumsum, creating a look up table by the formula we have been taught and updating the image based on that formula, eventually we creating a histogram of the updated image too (if the original image is RGB type, we will convert it back to RGB).
![gray_equalized](https://user-images.githubusercontent.com/63747865/161278448-3f9870b2-d2fe-4955-8198-e7db028263a2.png)
![rgb_equalized](https://user-images.githubusercontent.com/63747865/161278619-f3ec85c9-1a1b-4714-9c16-f6c26e309d34.png)

7. **quantizeImage** - This function gets an image, a number of colors (nQuant) and number of iterations (nIter), the function returns a list of the quantized images improvements and a list of the errors in each iteration.

Proccess: if the image is RGB type, then we first convert it to YIQ and performing the changes only on the Y channel. calculating the histogram, creating an initial borders by having a even parts of pixels in each border, calculating the weighted average in each border, updating the the image by the weighted average, updating the borders by the average of the weighted averages, adding the updated image and the error (MSE) from the original image and repeating this proccesss nIter of times (if the original image is RGB type, we will convert it back to RGB).
![2_quantized](https://user-images.githubusercontent.com/63747865/161281734-3f29547b-e004-4d76-b219-5ed57701fd49.png)
![4_quantized](https://user-images.githubusercontent.com/63747865/161285012-2130d197-875f-4e4f-b6b5-9eeace4d3b29.png)
![8_quantized](https://user-images.githubusercontent.com/63747865/161283480-f1f40d41-a833-4f8f-99aa-ed794209f2d0.png)
![64_quantized](https://user-images.githubusercontent.com/63747865/161283823-988fc026-1d83-4a62-b1a8-ae2c96de2a7a.png)
![128_quantized](https://user-images.githubusercontent.com/63747865/161283534-ffef330a-3c7f-4ff0-a8bb-351dcb58604e.png)

8. **gammaDisplay** - This function gets an image and a integer representation, the function displays the image based on the current gamma, when you can change the gamma and see how it affects the image itself.

Proccess: updating the image based on the current gamma by the formula we have been taught.

In this function i created another function named **emptyFunc** - this function has been created only because the function createTrackbar must get a function while im doing all of the proccess in the function gammaDisplay.

![gray_gamma](https://user-images.githubusercontent.com/63747865/161395078-e4515b24-9624-4d73-a865-eef85bc64be6.gif)
![rgb_gamma](https://user-images.githubusercontent.com/63747865/161395093-db1d5fd8-8310-45b4-ba0a-b08e0232511d.gif)

**Questions in section 4.5**

1. The reason the program will crash if we have a gray level segment with no pixels, it's because we suppose to divide by all of the pixels to calculate the weighted average, and by that will get a Zero Division Error and the program will crash.

2. The reason that we will have more colors in the RGB quantization it's because we are working only on the Y channel, and the I and Q channels are changing the colors too in some way.







