import { GUI } from 'lil-gui';

class Config {

  private gui: GUI;

  public renderingOptions = {
    mode: 0,
    filterSize: 32,
    particleRadius: 0.02,
    particleTickness: 0.25,
    tintColor: { r: 6, g: 105, b: 217 }
  };

  constructor() {
    this.gui = new GUI();
  }

  public initRenderingOptions(onChangeFunc: (msg) => void) {

    const renderingOptionGUI = this.gui.addFolder('Fluid Rendering Options');
    renderingOptionGUI.add(this.renderingOptions, 'filterSize', 0, 32).step(2);
    renderingOptionGUI.add(this.renderingOptions, 'mode', 
      { PBR: 0, 'PBR(No Refraction)': 1, Diffuse: 2, Normal: 3, Depth: 4, Thickness: 5 }
    );
    renderingOptionGUI.add(this.renderingOptions, 'particleRadius', 0.005, 0.05);
    renderingOptionGUI.add(this.renderingOptions, 'particleTickness', 0, 1.0);
    renderingOptionGUI.addColor(this.renderingOptions, 'tintColor', 255);
    renderingOptionGUI.onFinishChange(onChangeFunc);
    
  }

}

export { Config };