for /D %%s in (.\*) do (
    ECHO %%s
    cd %%s
    for /D %%s in (.\*) do (
        md %%s\frames
        ffmpeg -i %%s/cam1.mkv -r 20 -f image2 %%s/frames/cam1-%%05d.png
        ffmpeg -i %%s/cam2.mkv -r 20 -f image2 %%s/frames/cam2-%%05d.png
    )
    
    cd ".."
)

