import { GUI } from 'lil-gui';

class Config {

  private gui: GUI;

  public scnenOptions = {
    skybox: true,
    mesh: true,
    fluid: true
  }

  public renderingOptions = {
    mode: 0,
    filterSize: 32,
    particleRadius: 0.02,
    particleTickness: 0.25,
    tintColor: { r: 6, g: 105, b: 217 }
  };

  public simulationOptions = {
    XSPH: 0.01,
    vorticity: 0.1,
    surfaceTension: 0.5
  };

  constructor() {
    this.gui = new GUI();
  }

  public initSceneOptions(onChangeFunc: (msg) => void) {

    const sceneOptionGUI = this.gui.addFolder('Scene Options');
    sceneOptionGUI.add(this.scnenOptions, 'skybox');
    sceneOptionGUI.add(this.scnenOptions, 'mesh');
    sceneOptionGUI.add(this.scnenOptions, 'fluid');
    sceneOptionGUI.onFinishChange(onChangeFunc);

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

  public initSimulationOptions(onChangeFunc: (msg) => void) {

    const simulationOptionGUI = this.gui.addFolder('Fluid Simulation Options');
    simulationOptionGUI.add(this.simulationOptions, 'XSPH', 0, 0.1);
    simulationOptionGUI.add(this.simulationOptions, 'vorticity', 0, 1);
    simulationOptionGUI.add(this.simulationOptions, 'surfaceTension', 0, 1);
    simulationOptionGUI.onFinishChange(onChangeFunc);
    
  }

}

export { Config };