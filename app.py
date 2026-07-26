import gradio as gr
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from scanner import scan_dataset
from backup import create_backup
from protector import analyze_dataset
from captioner import caption_dataset

tacky_theme = gr.themes.Default(
    primary_hue="fuchsia",
    secondary_hue="cyan",
    neutral_hue="slate",
).set(
    button_primary_background_fill="*primary_500",
    button_primary_text_color="white",
    block_border_width="2px",
    block_border_color="*secondary_500"
)

def select_folder():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory()
    root.destroy()
    return folder_path

def scan_action(dataset_path):
    path = Path(dataset_path.strip())
    if not path.exists() or not path.is_dir():
        return "Error: Invalid directory path.", []
    
    results = scan_dataset(path)
    protection = analyze_dataset(path)
    
    report = (
        f"--- SCAN RESULTS ---\n"
        f"Total Images: {len(results['images'])}\n"
        f"Missing Captions: {len(results['missing_captions'])}\n\n"
        f"--- HEALTH CHECK ---\n"
        f"Corrupted Images: {len(protection['corrupted_images'])}\n"
    )
    return report, results['missing_captions']

def process_batch(dataset_path, caption_engine, subject_type, text_style, extra_features, trigger_word, custom_tags, output_format, do_backup, missing_images_state, progress=gr.Progress()):
    path = Path(dataset_path.strip())
    
    if not missing_images_state:
        return "No images to caption. Please scan a valid folder first."
        
    if do_backup:
        progress(0, desc="Creating backup...")
        create_backup(path)
        
    def update_progress(current, total, file_name):
        progress(current / total, desc=f"[{current}/{total}] Captioning {file_name}...")

    result = caption_dataset(
        missing_images_state, 
        caption_engine, 
        subject_type, 
        text_style, 
        trigger_word, 
        custom_tags,
        extra_features,
        output_format,
        progress_callback=update_progress
    )
    
    return f"Captioning Complete!\nGenerated: {result['created']}\nSkipped: {result['skipped']}\nFailed: {result['failed']}"

def load_gallery(dataset_path):
    path = Path(dataset_path.strip())
    if not path.exists() or not path.is_dir():
        return [], []
        
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [f.as_posix() for f in path.rglob("*") if f.suffix.lower() in extensions]
    return images, images

def get_caption(evt: gr.SelectData, image_paths):
    img_path = Path(image_paths[evt.index])
    txt_path = img_path.with_suffix(".txt")
    content = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
    return str(img_path), content

def save_caption(img_path, content):
    if not img_path:
        return "No image selected!"
    try:
        txt_path = Path(img_path).with_suffix(".txt")
        txt_path.write_text(content, encoding="utf-8")
        return f"✅ Saved successfully!"
    except Exception as e:
        return f"❌ Error saving: {str(e)}"

with gr.Blocks(title="Hyper Captioner") as ui:
    gr.Markdown("# 🚀 HYPER CAPTIONER UI")
    
    with gr.Tabs():
        with gr.Tab("Dataset Batch"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("### 📂 1. Source Directory")
                        with gr.Row():
                            dataset_input = gr.Textbox(label="Dataset Path", placeholder="C:/datasets...", scale=4)
                            folder_btn = gr.Button("📁 Browse", scale=1)
                            
                        scan_btn = gr.Button("🔍 Scan Dataset Directory", variant="secondary")
                        
                    with gr.Group():
                        gr.Markdown("### ⚙️ 2. Training Targets & Options")
                        caption_engine = gr.Radio(choices=["JoyCaption (4-bit)", "Florence-2 (Large)"], value="JoyCaption (4-bit)", label="AI Captioning Engine")
                        text_style = gr.Radio(choices=["Flux (Natural Language)", "SDXL (Tags)"], value="Flux (Natural Language)", label="Captioning Style")
                        subject_type = gr.Radio(choices=["General", "Character", "Clothing", "Style", "Pose"], value="General", label="Subject Focus")
                        
                        extra_features = gr.CheckboxGroup(
                            choices=["Camera Angle & Framing", "Lighting & Shadows", "Background & Environment"],
                            value=["Camera Angle & Framing", "Lighting & Shadows", "Background & Environment"],
                            label="Include Specific Details in Prompt"
                        )
                        
                    with gr.Group():
                        gr.Markdown("### ✍️ 3. Prompting & Output")
                        
                        output_format = gr.Radio(choices=["Sidecar (.txt)", "Master Metadata (.csv)", "Master Metadata (.json)"], value="Sidecar (.txt)", label="Output File Format")
                        
                        trigger_word = gr.Textbox(label="Trigger Word", placeholder="e.g. m1k4")
                        custom_tags = gr.Textbox(label="Custom Tags / Base Caption (Applied to all)", placeholder="e.g. masterpiece, high quality")
                        do_backup = gr.Checkbox(label="Create Backup Before Run", value=True)
                        
                    run_btn = gr.Button("🔥 START BATCH CAPTIONING 🔥", variant="primary", size="lg")
                    
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Status & Output")
                    scan_output = gr.Textbox(label="Scan Results", lines=6, interactive=False)
                    result_output = gr.Textbox(label="Batch Progress", lines=6, interactive=False)
            
            missing_images_state = gr.State([])
            
            folder_btn.click(fn=select_folder, outputs=[dataset_input])
            scan_btn.click(fn=scan_action, inputs=[dataset_input], outputs=[scan_output, missing_images_state])
            run_btn.click(
                fn=process_batch,
                inputs=[dataset_input, caption_engine, subject_type, text_style, extra_features, trigger_word, custom_tags, output_format, do_backup, missing_images_state],
                outputs=[result_output]
            )

        with gr.Tab("Preview & Edit"):
            gr.Markdown("### 🖼️ Interactive Gallery Reviewer")
            gr.Markdown("*Note: The gallery editor relies on Sidecar (.txt) files for on-the-fly editing.*")
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        preview_dataset_input = gr.Textbox(label="Dataset Path", placeholder="Select directory...", scale=3)
                        preview_folder_btn = gr.Button("📁 Browse", scale=1)
                        preview_load_btn = gr.Button("🔄 Load Images", variant="secondary", scale=1)
                    
                    gallery = gr.Gallery(label="Dataset Images", columns=4, height="650px", object_fit="contain")
                    gallery_state = gr.State([])
                    
                with gr.Column(scale=1):
                    selected_image = gr.Textbox(label="Selected Image Path", interactive=False)
                    caption_editor = gr.Textbox(label="Caption File (.txt)", lines=18)
                    save_btn = gr.Button("💾 Save Caption", variant="primary")
                    save_status = gr.Textbox(label="Status", interactive=False)
            
            preview_folder_btn.click(fn=select_folder, outputs=[preview_dataset_input])
            preview_load_btn.click(fn=load_gallery, inputs=[preview_dataset_input], outputs=[gallery, gallery_state])
            gallery.select(fn=get_caption, inputs=[gallery_state], outputs=[selected_image, caption_editor])
            save_btn.click(fn=save_caption, inputs=[selected_image, caption_editor], outputs=[save_status])

if __name__ == "__main__":
    ui.launch(
        inbrowser=True, 
        theme=tacky_theme, 
        allowed_paths=["C:/", "D:/", "E:/", "F:/"]
    )