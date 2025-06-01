# Ray-Tracer

# Ray Tracer Implementation with Blinn-Phong Shading

## Purpose

Presentation of the final version of the project for my portfolio. This project implements a ray tracer that renders scenes with spheres and planes, utilizing the Blinn-Phong shading model. **Its primary goal is to provide a clear and understandable implementation of the ray tracing algorithm to demonstrate my understanding of computer graphics principles.**

## Key Features

* **Supported Objects:** Renders scenes containing spheres and planes.
* **Lighting Model:** Implements the Blinn-Phong shading model, capturing diffuse, specular, and ambient lighting effects.
* **Light Source:** Supports a single point light source.
* **Vector Library:** Includes a custom 3D vector library (`Vec3d`) built using NumPy for efficient vector operations.
* **Soft Shadows:** Implements soft shadows, producing smoother transitions between light and shadow.
* **Ray Depth:** Allows for configurable ray tracing depth, enabling the simulation of reflections and refractions (if implemented).
* **Surface Properties:** Supports various surface properties typical of the Blinn-Phong model, including:
    * `colour`: Surface color (Vec3d)
    * `roughness`: Surface roughness
    * `reflectivity`: Surface reflectivity
    * `diffuse`: Diffuse lighting coefficient
    * `shininess`: Specular highlight shininess
    * `specular`: Specular lighting coefficient
    * `ambient`: Ambient lighting coefficient

## Technologies Used

* Programming Language: Python
* Libraries: NumPy, random, time, matplotlib

## Performance Considerations

**This ray tracer is primarily an educational project, focusing on code clarity and algorithmic understanding. As such, it may not be optimized for performance, and rendering times can be significant, especially for complex scenes.** The emphasis is on demonstrating the core concepts of ray tracing rather than achieving real-time rendering speeds. However, this allowed for a deeper exploration of the underlying mathematics and physics of light and rendering.

## Setup Instructions

1.  Make sure you have Python 3.x installed.
2.  Install the required libraries if you haven't already:

    ```bash
    pip install numpy matplotlib
    ```
3.  Ensure that all the project files (including your vector library implementation) are in the same directory.

## Running the Code

To render a scene, simply run the main Python script. The script should be executable directly and does not require any additional setup. The output will be displayed using Matplotlib.

## Example Renders

The repository includes example rendered images to showcase the capabilities of the ray tracer. These images demonstrate the effects of Blinn-Phong shading, soft shadows, and various surface properties.

## Additional Information

This project provides a detailed implementation of a ray tracer, **offering a valuable learning experience in computer graphics rendering techniques.** While performance was not the primary concern, the code is structured to be readable and understandable, making it easier to grasp the fundamental principles.

The inclusion of soft shadows and configurable ray depth adds to the realism of the rendered scenes and showcases a more in-depth exploration of the topic.
