using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.IO.Pipes;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.Scripting;

public class FireSimulation : MonoBehaviour
{
    private const float FIRESPREAD_SIZE_RAD = 5;
    private const float FIRESPREAD_RAD_SQR = FIRESPREAD_SIZE_RAD * FIRESPREAD_SIZE_RAD;
    private const int NUMBER_OF_SIMULATIONS_PER_FRAME = 200;

    GameManager gameManager;

    QuadTree<IFlamable> quadTree;
    List<IFlamable> activeFire;

    Rect fireRect;
    Vector3 windVector;
    Coroutine fireCacheRecalculateCoroutine;

    int FileCounter = 0;
    private Light light;
    // private byte[] Bytes;
    // private Texture2D Image;
    private List<string> cameras = new List<string>();
    // Windows
    private NamedPipeServerStream producerPipe;

    // // Linux
    // private FileStream producerPipe;
    private string dataPath;
    // Use this for initialization
    void Awake()
    {
        // GarbageCollector.GCMode = GarbageCollector.Mode.Manual;
        Terrain activeTerrain = Terrain.activeTerrain;
        gameManager = GameObject.FindGameObjectWithTag("GameController").GetComponent<GameManager>();

        quadTree = new QuadTree<IFlamable>(100, new Rect(-FIRESPREAD_SIZE_RAD * 2.1f, -FIRESPREAD_SIZE_RAD * 2.1f, activeTerrain.terrainData.size.x + FIRESPREAD_SIZE_RAD * 2.1f, activeTerrain.terrainData.size.z + FIRESPREAD_SIZE_RAD * 2.1f));
        activeFire = new List<IFlamable>(1000);

        gameManager.windDirectionChanged += gameManager_windDirectionChanged;
        gameManager.windSpeedChanged += gameManager_windSpeedChanged;

        RecalculateWindVector();

        fireRect = new Rect(0, 0, FIRESPREAD_SIZE_RAD * 2, FIRESPREAD_SIZE_RAD * 2);
        enabled = false;

        Vector3 pos = new Vector3(10,50,10);
        Camera modelCam = GameObject.Find("snapshot_camera").GetComponent<Camera>();
        light = GameObject.Find("Directional Light").GetComponent<Light>();
        light.cullingMask = 0;

        cameras.Add("Full");
        for(int i=0;i<24;i++){
            for(int j=0;j<24;j++){
                GameObject CamGo = new GameObject();
                CamGo.name= "Cam_"+i.ToString()+"_"+j.ToString();
                Camera cam = CamGo.AddComponent<Camera>();
                cam.CopyFrom(modelCam);
                // Debug.Log(pos);
                CamGo.transform.position = pos;
                cam.enabled = false;
                cameras.Add(cam.name);
                pos[0]+=20;
            }
            pos[2]+=20;
            pos[0]=10;
        }
        light = GameObject.Find("Directional Light").GetComponent<Light>();
        dataPath =  Application.dataPath+"/"+System.DateTime.Now.ToString("yyyy-MM-dd-HH-mm-ss")+"/";
        if (File.Exists(dataPath) == false){
            Directory.CreateDirectory(dataPath);
        }


        // // Windows start
        // producerPipe = new NamedPipeServerStream("PipesOfPiece", direction: PipeDirection.InOut,maxNumberOfServerInstances:10);
        // Debug.Log("Waiting for connection...");
        // producerPipe.WaitForConnection();
        // Debug.Log("Connected!");
        // // Windows end


        // // Linux start
        // Debug.Log("Waiting for connection...");
        // producerPipe = File.OpenWrite("/home/umang/big_data.fifo");
        // Debug.Log("Connected!");
        // // Linux end
        StartCoroutine(Simulate());
    }

    // void FixedUpdate(){
    //     StartCoroutine(Simulate());
    // }

    void gameManager_windSpeedChanged(float obj)
    {
        if (fireCacheRecalculateCoroutine != null)
        {
            StopCoroutine(fireCacheRecalculateCoroutine);
            fireCacheRecalculateCoroutine = null;
        }
        RecalculateWindVector();
        fireCacheRecalculateCoroutine = StartCoroutine(RecalculateFiresCache(windVector));
    }

    void gameManager_windDirectionChanged(float obj)
    {
        if (fireCacheRecalculateCoroutine != null)
        {
            StopCoroutine(fireCacheRecalculateCoroutine);
            fireCacheRecalculateCoroutine = null;
        }
        RecalculateWindVector();
        fireCacheRecalculateCoroutine = StartCoroutine(RecalculateFiresCache(windVector));
    }

    private void RecalculateWindVector()
    {
        windVector = new Vector3(Mathf.Cos(Mathf.Deg2Rad * gameManager.windDirection), 0, Mathf.Sin(Mathf.Deg2Rad * gameManager.windDirection)) * gameManager.windSpeed;
    }

    void OnDrawGizmos()
    {
        if (quadTree != null)
        {
            quadTree.DrawDebug();
        }
    }

    IEnumerator Simulate()
    {
        int iterations = 0;
        while (true)
        {
            if (!enabled)
            {
                yield return null;
                continue;
            }
            
            if (iterations%500 == 0){
                light.cullingMask = -1;
                yield return new WaitForFixedUpdate();
                yield return new WaitForFixedUpdate();
                List<Coroutine> runningCoroutines = new List<Coroutine>();
                Time.timeScale = 0;
                Debug.Log("Start: "+System.DateTime.UtcNow.ToString("HH:mm:ss"));
                int i=0;
                List<string> cam_batch = new List<string>();
                foreach (string camName in cameras){
                    i++;
                    cam_batch.Add(camName);
                    if(i%10==0){
                        runningCoroutines.Add(StartCoroutine(Capture(cam_batch)));
                        cam_batch = new List<string>();
                    }
                    // runningCoroutines.Add(StartCoroutine(Capture(camName)));
                }
                runningCoroutines.Add(StartCoroutine(Capture(cam_batch)));
                foreach(Coroutine c in runningCoroutines)
                {
                    yield return c;
                }
                Time.timeScale = 1;
                light.cullingMask = 0;
                Debug.Log(FileCounter);
                FileCounter++;
                Debug.Log(System.DateTime.UtcNow.ToString("HH:mm:ss"));
                // System.GC.Collect(2);
            }
            iterations+=1;
            // yield return new WaitForFixedUpdate();
            for (int i = 0; i < activeFire.Count; i++)
            {
                
                IFlamable fire = activeFire[i];
                List<IFlamable> fireCache = fire.GetCacheList();
                if (fireCache.Count == 0)
                    continue;
                Vector2 position = fire.GetPosition();
                ArrangeRect(ref fireRect, ref position, ref windVector);
                bool atLeastOneHeating = false;
                for (int j = 0; j < fireCache.Count; j++)
                {
                    IFlamable flammable = fireCache[j];
                    FlameState flameState = flammable.GetFlameState();

                    //keep out invalid
                    if (flammable == null || flameState == FlameState.BURNED_OUT || flameState == FlameState.BURNING)
                        continue;

                    atLeastOneHeating = true;
                    if (flammable.GetFlameState() != FlameState.HEATING)
                        flammable.SetFlameState(FlameState.HEATING);

                    flammable.AddTemperature(Random.value + 0.5f);
                }
                if (!atLeastOneHeating)
                    fireCache.Clear();
            }
           
            yield return null;
        }
    }

    private IEnumerator Capture(List<string> camBatch)
    {
        foreach (string camName in camBatch){

            Camera Cam = GameObject.Find(camName).GetComponent<Camera>();

            Cam.Render();

            RenderTexture.active = Cam.targetTexture;
            Texture2D Image = new Texture2D(Cam.targetTexture.width, Cam.targetTexture.height);
            Image.ReadPixels(new Rect(0, 0, Cam.targetTexture.width, Cam.targetTexture.height), 0, 0);
            Image.Apply();


            byte[] Bytes = Image.EncodeToPNG();

            var dir_name = Cam.name;
            // // PIPE IMPLEMENTATION
            // var dir_name = Cam.name+"/"+FileCounter+".png";
            // string encodedBytes = System.Convert.ToBase64String(Bytes);
            // // Debug.Log("Writing to pipe");
            // producerPipe.Write(System.Text.Encoding.ASCII.GetBytes(dir_name), 0, dir_name.Length);
            // producerPipe.WriteByte(System.Text.Encoding.ASCII.GetBytes("\n")[0]);
            
            // producerPipe.Write(System.Text.Encoding.ASCII.GetBytes(encodedBytes), 0, encodedBytes.Length);
            // producerPipe.WriteByte(System.Text.Encoding.ASCII.GetBytes("\n")[0]);
            // Debug.Log("Written to pipe");
            // // PIP IMPLEMENTATION END
            if (Directory.Exists(dataPath + dir_name + "/") == false)
                Directory.CreateDirectory(dataPath + dir_name + "/");
            File.WriteAllBytes(dataPath + dir_name + "/" + FileCounter + ".png", Bytes);
        }
        yield return null;
    }

    private void ArrangeRect(ref Rect rect, ref Vector2 position, ref Vector3 windVect)
    {
        rect.x = position.x - (1 - windVect.x) * FIRESPREAD_SIZE_RAD;
        rect.y = position.y - (1 + windVect.z) * FIRESPREAD_SIZE_RAD;
    }

    public void AddTree(IFlamable tree)
    {
        if (fireCacheRecalculateCoroutine != null)
        {
            StopCoroutine(fireCacheRecalculateCoroutine);
            fireCacheRecalculateCoroutine = null;
        }
        quadTree.Insert(tree);
        fireCacheRecalculateCoroutine = StartCoroutine(RecalculateFiresCache(windVector));
    }

    public void RemoveTree(IFlamable tree)
    {
        quadTree.Remove(tree);
    }

    public void ClearSimulation()
    {
        quadTree.Clear();
        activeFire.Clear();
    }

    public void SetSimulationActive(bool isSimulating)
    {
        enabled = isSimulating;
    }

    public void AddFire(IFlamable fire)
    {
        activeFire.Add(fire);
        RecalculateFireCache(fire, windVector);
    }

    public void RemoveFire(IFlamable position)
    {
        activeFire.Remove(position);
    }

    private IEnumerator RecalculateFiresCache(Vector3 windVector)
    {
        for (int i = 0; i < activeFire.Count; i++)
        {
            IFlamable fire = activeFire[i];
            RecalculateFireCache(fire, windVector);
        }
        yield return null;
    }

    private void RecalculateFireCache(IFlamable fire, Vector3 windVector)
    {
        Vector2 position = fire.GetPosition();
        Rect r = new Rect(0, 0, FIRESPREAD_SIZE_RAD * 2, FIRESPREAD_SIZE_RAD * 2);
        ArrangeRect(ref r, ref position, ref windVector);
        Vector2 fireRectCenter = r.center;
        List<IFlamable> tmp = fire.GetCacheList();
        tmp.Clear();
        quadTree.Retrieve(ref tmp, ref r);
        //make circle from rect
        tmp.RemoveAll(x =>
        {
            return (x.GetPosition() - fireRectCenter).sqrMagnitude > FIRESPREAD_RAD_SQR;
        });
    }
}
