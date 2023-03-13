# onnxruntime-azure-function-FEL-object-detection

Inspiration from: https://www.youtube.com/watch?v=MCafgeqWMhQ&ab_channel=ONNXRuntime

Small specification to run project:
	- Same as in video

Open project in editor - I used VS Code 

Press F5 to run in debug mode
	- May need to download "azure-functions-core-tools" From https://github.com/Azure/azure-functions-core-tools#installing
Or "Ctrl + F5" to just run

This will give you a link to a local host link: 
	- In my case it was: "FELObjecDetectionHttpTrigger: [GET,POST] http://localhost:7071/api/FELObjecDetectionHttpTrigger"

Now use Postman or some other tool to send post request to the link above
	- I have some examples of images here: https://documenter.getpostman.com/view/5572934/2s93JowkfR 
	- The same examples are also in the folder "examples_images" as .txt files
