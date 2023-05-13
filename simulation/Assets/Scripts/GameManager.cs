using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using System.IO;
public enum ButtonAction
{
    GENERATE,
    CLEAR,
    SIMULATE,
    FIRE,
    QUIT
}

public enum ModeAction
{
    ADD,
    REMOVE,
    TOGGLE
}

public class GameManager : MonoBehaviour
{

    private const int NUMBER_OF_TREE_INSTANCES = 20000;
    private const int NUMBER_OF_TREE_GENERATED_PER_FRAME = 15000;
    private const int NUMBER_OF_RANDOM_FIRES = 1;

    public event Action<bool> simulationStateChanged;
    public event Action<float> windSpeedChanged;
    public event Action<float> windDirectionChanged;

    public GameObject firepitPrefab;
    public GameObject treePrefab;
    public GameObject treeColliderPrefab;

    private Terrain activeTerrain;
    private FireSimulation fireSim;
    private Light light;
    bool isSimulating;
    bool isGenerating;
    ModeAction actionMode;
    public float windSpeed
    {
        get;
        private set;
    }
    public float windDirection
    {
        get;
        private set;
    }
    // public int FileCounter = 0;
    // private byte[] Bytes;
    // private Texture2D Image;
    // private List<Camera> cameras = new List<Camera>();
    // Use this for initialization
    void Start()
    {
        isSimulating = false;
        isGenerating = false;
        actionMode = ModeAction.ADD;
        windDirection = 0.5f;
        windSpeed = 0.5f;
        activeTerrain = Terrain.activeTerrain;
        fireSim = activeTerrain.GetComponent<FireSimulation>();
        ClearMap();
        Scene scene = gameObject.scene;
        fireSim.SetSimulationActive(isSimulating);
        // Vector3 pos = new Vector3(10,50,10);
        // Camera modelCam = GameObject.Find("snapshot_camera").GetComponent<Camera>();
        light = GameObject.Find("Directional Light").GetComponent<Light>();
        light.cullingMask = 0;
        // cameras.Add(GameObject.Find("Full").GetComponent<Camera>());
        // for(int i=0;i<24;i++){
        //     for(int j=0;j<24;j++){
        //         GameObject CamGo = new GameObject();
        //         CamGo.name= "Cam_"+i.ToString()+"_"+j.ToString();
        //         Camera cam = CamGo.AddComponent<Camera>();
        //         cam.CopyFrom(modelCam);
        //         // Debug.Log(pos);
        //         CamGo.transform.position = pos;
        //         cam.enabled = false;
        //         cameras.Add(cam);
        //         pos[0]+=20;
        //     }
        //     pos[2]+=20;
        //     pos[0]=10;
        // }
        
        
        OnButtonClicked(0);
        OnButtonClicked(3);
        StartCoroutine(startAnimation());
        
    }

    void OnApplicationQuit()
    {
        ClearMap();
    }
    IEnumerator startAnimation() {
        yield return new WaitForSeconds(5);
        OnButtonClicked(2);
    }
    // Update is called once per frame
    void FixedUpdate()
    {
        // if (Input.GetMouseButtonDown(0) && !EventSystem.current.IsPointerOverGameObject())
        // {
        //     Ray inputRay = Camera.main.ScreenPointToRay(Input.mousePosition);
        //     RaycastHit hit;
        //     if (Physics.Raycast(inputRay, out hit))
        //     {
        //         TerrainData td = activeTerrain.terrainData;
        //         switch (actionMode)
        //         {
        //             case ModeAction.ADD:
        //                 Vector3 hitPoint = hit.point;
        //                 activeTerrain.AddTreeInstance(CreateTreeInstance(hitPoint.x / td.size.x, hitPoint.z / td.size.z));
        //                 break;
        //             case ModeAction.REMOVE:
        //                 IDeletable tree = hit.collider.gameObject.GetComponent<IDeletable>();
        //                 if (tree != null && hit.collider.gameObject.tag == "Tree")
        //                 {
        //                     tree.Delete();
        //                 }
        //                 break;
        //             case ModeAction.TOGGLE:
        //                 if (hit.collider.gameObject.tag == "Tree")
        //                 {
        //                     //Fire fire = hit.collider.GetComponentInChildren<Fire>();
        //                     IFlamable flammable = hit.collider.GetComponent<IFlamable>();
        //                     if (flammable == null)
        //                         break;
        //                     if (flammable.GetFlameState() == FlameState.BURNING)
        //                     {
        //                         flammable.SetFlameState(FlameState.NONE);
        //                     }
        //                     else
        //                     {
        //                         flammable.SetFlameState(FlameState.BURNING);
        //                     }
        //                     break;
        //                 }
        //                 if (hit.collider.gameObject.tag == "Fire")
        //                 {
        //                     IDeletable firepit = hit.collider.gameObject.GetComponent<IDeletable>();
        //                     if (firepit != null)
        //                     {
        //                         firepit.Delete();
        //                     }
        //                     break;
        //                 }
        //                 Instantiate(firepitPrefab, hit.point, Quaternion.identity, activeTerrain.transform);
        //                 break;
        //             default:
        //                 break;
        //         }
        //     }
        // }
        // if((Time.frameCount%1000)%450 == 0){
        //     light.cullingMask = -1;
        // }
        // if ((Time.frameCount%1000) % 500 == 0)
        // {
        //     // if(light.enabled)
        //     light.cullingMask = -1;
        //     Debug.Log("Start: "+System.DateTime.UtcNow.ToString("HH:mm:ss"));
        //     // StartCoroutine(Capture());
            
        //     // GameObject camGO = GameObject.Find("snapshot_camera");
        //     // Camera Cam = camGO.GetComponent<Camera>();
    
        //     // RenderTexture currentRT = RenderTexture.active;
        //     // RenderTexture.active = Cam.targetTexture;
        //     // wait();
            
        //     // foreach (Camera Cam in cameras){
        //     //     Cam.enabled = true;
        //     //     Cam.Render();
        //     //     // Debug.Log(Cam.targetTexture);
        //     //     RenderTexture currentRT = RenderTexture.active;
        //     //     RenderTexture.active = Cam.targetTexture;
        //     //     Image = new Texture2D(cameras[0].targetTexture.width, cameras[0].targetTexture.height);
        //     //     Image.ReadPixels(new Rect(0, 0, Cam.targetTexture.width, Cam.targetTexture.height), 0, 0);
        //     //     Image.Apply();
        //     //     RenderTexture.active = currentRT;
        
        //     //     Bytes = Image.EncodeToPNG();
        //     //     Destroy(Image);
        //     //     String dir_name = Cam.gameObject.name;
        //     //     if (Directory.Exists("C:\\Big Data\\project\\"+dir_name+"/") == false)
        //     //         Directory.CreateDirectory("C:\\Big Data\\project\\"+dir_name+"/");
        //     //     File.WriteAllBytes("C:\\Big Data\\project\\"+dir_name+"/" + FileCounter + ".png", Bytes);
        //     //     Cam.enabled = false;
        //     // }
        //     // GC.Collect();
        //     // FileCounter++;
        // }

    }

    // private IEnumerator Capture()
    // {

    //     Vector3 pos = new Vector3(10, 50, 10);
    //     // Camera modelCam = GameObject.Find("snapshot_camera").GetComponent<Camera>();
        
    //     // var path = GameObject.Find("Path").GetComponent<InputField>().text;
    //     // GameObject.Find("Directional Light").disable = false;
    //     Time.timeScale = 0;
    //     foreach (Camera Cam in cameras){
    //             // GameObject CamGo = new GameObject();
    //             // CamGo.name= "Cam_"+i.ToString()+"_"+j.ToString();
    //             // Camera Cam = CamGo.AddComponent<Camera>();
    //             // Cam.CopyFrom(modelCam);
    //             // Debug.Log(pos);
    //             // CamGo.transform.position = pos;
    //             // cameras.Add(cam);
    //             // pos[0] += 20;
    //             Cam.Render();
    //             // Debug.Log(Cam.targetTexture);
    //             // RenderTexture currentRT = RenderTexture.active;
    //             RenderTexture.active = Cam.targetTexture;
    //             Image = new Texture2D(Cam.targetTexture.width, Cam.targetTexture.height);
    //             Image.ReadPixels(new Rect(0, 0, Cam.targetTexture.width, Cam.targetTexture.height), 0, 0);
    //             Image.Apply();
    //             // RenderTexture.active = currentRT;

    //             Bytes = Image.EncodeToPNG();
    //             Destroy(Image);
    //             String dir_name = Cam.name;
    //             // "C:\\Big Data\\project\\"
    //             if (Directory.Exists("C:/Big Data/project/"+ dir_name + "/") == false)
    //                 Directory.CreateDirectory("C:/Big Data/project/"+ dir_name + "/");
    //             File.WriteAllBytes("C:/Big Data/project/"+ dir_name + "/" + FileCounter + ".png", Bytes);
    //             //     Cam.enabled = false;
    //             // Destroy(CamGo);
    //     }
    //     Debug.Log(FileCounter);
    //     FileCounter++;
    //     Debug.Log(System.DateTime.UtcNow.ToString("HH:mm:ss"));
    //     // GameObject.Find("Directional Light").disable = true;
    //     // light.enabled = false;
    //     light.cullingMask = 0;
    //     Time.timeScale = 1;
    //     yield return null;
    // }

    public void OnButtonClicked(int action)
    {
        switch ((ButtonAction)action)
        {
            case ButtonAction.GENERATE:
                if (!isGenerating)
                {
                    isGenerating = true;
                    StartCoroutine(GenerateMap());
                }
                isGenerating = false;
                break;
            case ButtonAction.CLEAR:
                ClearMap();
                break;
            case ButtonAction.SIMULATE:
                if (isSimulating)
                    StopSimulation();
                else{
                    StartSimulation();
                }
                   
                if (simulationStateChanged != null)
                    simulationStateChanged(isSimulating);
                break;
            case ButtonAction.FIRE:
                StartRandomFire();
                break;
            case ButtonAction.QUIT:
                QuitApp();
                break;
            default:
                break;
        }
    }

    public void OnWindSpeedChanged(float speed)
    {
        windSpeed = speed;
        if (windSpeedChanged != null)
            windSpeedChanged(windSpeed);
    }

    public void OnWindDirectionChanged(float dir)
    {
        windDirection = dir;
        if (windDirectionChanged != null)
            windDirectionChanged(windDirection);
    }

    public void OnModeChanged(int mode)
    {
        actionMode = (ModeAction)mode;
    }

    private IEnumerator GenerateMap()
    {
        ClearMap();

        activeTerrain.drawTreesAndFoliage = false;
        for (int i = 0; i < NUMBER_OF_TREE_INSTANCES; i++)
        {
            activeTerrain.AddTreeInstance(CreateTreeInstance(UnityEngine.Random.value, UnityEngine.Random.value));
            if (i % NUMBER_OF_TREE_GENERATED_PER_FRAME == 0)
                yield return null;
        }
        activeTerrain.drawTreesAndFoliage = true;
        activeTerrain.Flush();
    }

    private TreeInstance CreateTreeInstance(float xPos, float zPos)
    {
        TreeInstance tree = new TreeInstance();
        Vector3 position = new Vector3(xPos, 0, zPos);
        tree.color = Color.green;
        tree.position = position;
        tree.prototypeIndex = 0;
        tree.rotation = 0;
        tree.widthScale = 3;
        tree.heightScale = 1;
        tree.lightmapColor = Color.white;

        position = Vector3.Scale(position, activeTerrain.terrainData.size);
        position.y = activeTerrain.SampleHeight(position);

        GameObject colliderGO = Instantiate(treeColliderPrefab, position, Quaternion.identity, activeTerrain.transform);
        colliderGO.name = "Tree_" + activeTerrain.terrainData.treeInstanceCount;
        colliderGO.GetComponent<MyTree>().terrainIndex = activeTerrain.terrainData.treeInstanceCount;

        return tree;
    }

    private void ClearMap()
    {
        activeTerrain.terrainData.treeInstances = new TreeInstance[0];
        Collider[] treeColliders = activeTerrain.gameObject.GetComponentsInChildren<Collider>();
        for (int i = 0; i < treeColliders.Length; i++)
        {
            Collider go = treeColliders[i];
            if (go.tag == "Tree")
            {
                GameObject.Destroy(go.gameObject);
            }
            if (go.tag == "Fire")
            {
                GameObject.Destroy(go.gameObject);
            }
        }
        fireSim.ClearSimulation();
        activeTerrain.Flush();
    }

    private void StartSimulation()
    {
        isSimulating = true;
        // Debug.Log(isSimulating);
        fireSim.SetSimulationActive(isSimulating);
    }

    private void StopSimulation()
    {
        isSimulating = false;
        fireSim.SetSimulationActive(isSimulating);
    }

    private void StartRandomFire()
    {
        for (int i = 0; i < NUMBER_OF_RANDOM_FIRES; i++)
        {
            int randomtreeIndex = UnityEngine.Random.Range(0, activeTerrain.terrainData.treeInstances.Length);
            GameObject.Find("Tree_" + randomtreeIndex).GetComponent<MyTree>().firstBurned = true;
            Vector3 position = Vector3.Scale(activeTerrain.terrainData.treeInstances[randomtreeIndex].position, activeTerrain.terrainData.size);
            Debug.Log(position);
            // Vector3 position = Vector3.Scale(new Vector3(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value), activeTerrain.terrainData.size);
            position.y = activeTerrain.SampleHeight(position);
            Instantiate(firepitPrefab, position, Quaternion.identity, activeTerrain.transform);
        }
    }

    private void QuitApp()
    {
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
         Application.Quit();
#endif
    }
}
