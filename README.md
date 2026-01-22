<h2 align="center"> ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment </h2>

<h3 align="center"> Anonymous authors </h3>

<img src="figure/overview.jpg" align="center" width="1000" height="450">
Recent advancements in text-guided audio generation have yielded promising results in diverse domains, including sound effects, environmental audio, speech, and music. However, jointly generating speech with environmental audio remains challenging due to the inherent disparities in their acoustic patterns and temporal dynamics. We propose ImmersiveTTS, an Environment-Aware text-to-speech (TTS) model that generates natural speech seamlessly integrated within environmental contexts by explicitly modeling cross-modal interactions. Our model builds on a multimodal diffusion transformer and fuses transcript-aligned speech latent with text-conditioned environmental context via joint attention. To enhance semantic consistency, we introduce a domain-specific representation alignment objective tailored to Environment-Aware TTS, leveraging complementary self-supervised representations from speech and audio encoders. Experimental results show that ImmersiveTTS achieves higher naturalness, intelligibility, and audio fidelity than existing approaches across objective metrics and human listening tests.

### Clone our repository
```
git clone https://github.com/immersivetts/ImmersiveTTS.git
cd ImmersiveTTS
```
