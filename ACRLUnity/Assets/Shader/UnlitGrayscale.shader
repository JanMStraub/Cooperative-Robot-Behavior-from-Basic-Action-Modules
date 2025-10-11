// Unlit shader that converts a texture to grayscale.
// This is significantly faster than CPU-based methods.
Shader "Unlit/Grayscale"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // Sample the original texture color.
                fixed4 col = tex2D(_MainTex, i.uv);
                
                // Calculate luminance using the standard formula (dot product).
                // This is faster and more accurate than a simple average.
                float grayscale = dot(col.rgb, float3(0.299, 0.587, 0.114));

                // Return the new grayscale color.
                return fixed4(grayscale, grayscale, grayscale, col.a);
            }
            ENDCG
        }
    }
}