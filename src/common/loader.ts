import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader';

class Loader {

  private loaderGLTF: GLTFLoader;
  private loaderFBX: FBXLoader;
  private loaderTexture: THREE.TextureLoader;
  private loaderCubeTexture: THREE.CubeTextureLoader;

  constructor() {

    this.loaderGLTF = new GLTFLoader();
    this.loaderFBX = new FBXLoader();
    this.loaderTexture = new THREE.TextureLoader();
    this.loaderCubeTexture = new THREE.CubeTextureLoader();

  }

  public loadGLTF(path: string) {

    return new Promise((
      resolve: (gltf: any) => void, 
      reject: (reason: any) => void
    ) => { 
      this.loaderGLTF.load( 
        path, 
        gltf => { resolve( gltf ); }, // onLoad
        null, // onProgress
        error => reject(error) // onError
      );
    });

  }

  public loadFBX(path: string) {

    return new Promise((
      resolve: (gltf: any) => void, 
      reject: (reason: any) => void
    ) => { 
      this.loaderFBX.load( 
        path, 
        gltf => { resolve( gltf ); }, // onLoad
        null, // onProgress
        error => reject(error) // onError
      );
    });

  }

  public loadTexture(path: string) {

    return new Promise((
      resolve: (texture: THREE.Texture) => void, 
      reject: (reason: any) => void
    ) => {
      this.loaderTexture.load(
        path,
        texture => { // onLoad
          texture.flipY = false; // flipY default is true
          resolve(texture);
        }, 
        null, // onProgress
        error => reject(error) // onError
      )
    });

  }

  public loadCubeTexture(paths: string[]) {

    return new Promise((
      resolve: (cubeTexture: THREE.CubeTexture) => void,
      reject: (reason: any) => void
    ) => {
      if (paths.length != 6) reject(new Error('Number of cube texture paths is Not 6'));
      this.loaderCubeTexture.load(
        paths,
        texture => resolve(texture),  // onLoad
        null, // onProgress
        error => reject(error) // onError
      )
    });

  }

}

let loader = new Loader();

export { loader };