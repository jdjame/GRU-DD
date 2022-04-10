# GRU-DD
-------
This repository serves as the codebase for the model downscaling model described in my thesis

### Abstract:
Global climate models represent major climate system components of the planet in order to generate long term, sparse, accurate realizations of future climatic events across the entire globe. Downscaling is the method by which these low resolution realizations are converted into high resolution simulations of climate events which can then be used by stakeholders and policy makers. 

Regional climate models dynamically downscale simulated climate by conditioning global climate models on  location-specific physical processes. Although these models are robust and reliable, they are computationally expensive when compared to statistical approaches for modeling a general relationship between global climate behaviour and local climate behavior. Therefore, there is need for downscaling methods that leverage the computational efficiency of statistical models while maintaining the performance of regional climate models.

In this thesis, we build upon previously proposed deep learning methods for dynamical downscaling through estimation of a regional climate model. Our proposed model is a generative adversarial network that leverages the effects of temporal dependencies within spatio-temporal climate events. 

### Model Structure

<table>
    <tr>
        <th >Critic</th>
        <th >Generator</th>
    </tr>
    <tr>
        <td> 
            <img src='./imgs/drdd critic.png'  alt="1" width = 360px height = 640px >
        </td>
        <td> 
            <img src='./imgs/drdd gen.png'  alt="2" width = 360px height = 640px >
        </td>
   </tr> 
</table>

### Data

For data, please refer to article: [Fast and accurate learned multiresolution dynamical downscaling for precipitation](https://arxiv.org/abs/2101.06813). The authors make the data, processing and modeling code available at their [github repository](https://github.com/lzhengchun/dsgan).

### In this repo 

In this model is the code to build and train the DRDD model described in the thesis. The model expects data shaped into Batch size $\times$ Time horizon $\times$ 1 $\times$ Height $\times$ Width. For low resolution images, height $\times$ width = 64 $\times$ 126. For high resolution images, height $\times$ width = 256 $\times$ 512.