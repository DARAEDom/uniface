async function select_db() {
  let x = document.querySelector('input[name="dbs"]:checked').id;
  if (x == null) {
    alert("No DB selected")
    console.log("No db selected");
  }
  try {
    $.ajax({
      type: 'POST',
      headers: {"Content-Type":"application/json;charset=UTF-8"},
      url: "/API/db",
      data: JSON.stringify({
        "opt":x
      })
    })
    .done(function allocate_db(data) {
      const img_menu = document.getElementById("image_menu");
      const tiles = document.getElementById("image_menu");
      let nodes = tiles.childNodes;
      if (nodes[2]) {
        nodes = Array.from(nodes);
        nodes.forEach(element => {
          element.remove();
        });
      }
      let ii = 0;
      for (let [key, value] of Object.entries(data)) {
        // CREATE RADIO BUTTON
        let radio_button = document.createElement("input")
        radio_button.style.display = "none";
        radio_button.setAttribute("id", "group" + ii);
        radio_button.setAttribute("type", "radio");
        radio_button.setAttribute("name", "group");
        img_menu.appendChild(radio_button);
        // FIND BUTTON AND ASSIGN FUNCTION
        const radio_assign = document.getElementById("group" + ii);
        radio_assign.onclick = function() {(`clicked group ${ii}`)}
        // CREATE LABEL FOR THE BUTTON
        let label = document.createElement("label");
        label.setAttribute("for", `group${ii}`);
        label.setAttribute("id", `label_group${ii}`);
        img_menu.appendChild(label);
        const label_assign = document.getElementById(`label_group${ii}`);
        // CREATE IMAGE
        let element = document.createElement("img");
        element.setAttribute("src", value[0]); 
        element.setAttribute("style", `left: ${30}px; top: ${40 + ii * 149}px;`);
        element.setAttribute("class", "image_frame");
        element.setAttribute("id", `group`+ii);
        label_assign.appendChild(element);
        const img_assign = document.getElementById(`group`+ ii);
        img_assign.onclick = function() {
          select_group(element);
        }
        ii++;
      } 
    });
  } catch (e) {
    alert(`Error has occured ${e}`);
  }
}

async function select_group(group){
  group = group.getAttribute("src");
  console.log(group);
  try {
    $.ajax({
      type: "POST",
      headers: {"Content-Type":"application/json;charset=UTF-8"},
      url: "/API/group",
      data: JSON.stringify({
        "group": group,
      })
    })
    .done(function allocate_images(data) {
      if (data.error) {
        console.log("Error", data);
        $('#').text(data.error).show();
      }
      else {
        const tiles = document.getElementById("image_display");
        let nodes = tiles.childNodes;
        console.log("Nodes:", nodes);
        if (nodes.firstChild) {
          nodes = Array.from(nodes);
          nodes.forEach(element => {
            element.removeChild();
            element.remove();
          });
        /*  for(let i=0; i<data.length; i++) {
            nodes[i].remove();
          }*/
        }
         let row = 0;
        let column = 0;
        for (let i=0; i<data.length; i++) {
          if (i % 5 == 0 && i != 0) {
            row++;
            column=0;
          }
          const img_display = document.getElementById("image_display");
          // CREATE RADIO BUTTON TO SELECT THE IMAGE
          let radio_button = document.createElement("input");
          radio_button.style.display = "none";
          radio_button.setAttribute("id", "image" + i);
          radio_button.setAttribute("type", "radio");
          radio_button.setAttribute("name", "image");
          img_display.appendChild(radio_button);
          // FIND ID OF CREATED RADIO BUTTON
          const radio_assign = document.getElementById("image" + i);
          radio_assign.onclick = function() {console.log(`clicked image ${i}`)}
          // CREATE LABEL FOR THE BUTTON
          let label = document.createElement("label");
          label.setAttribute("for", `image${i}`);
          label.setAttribute("id", `label${i}`);
          img_display.appendChild(label);
          const label_assign = document.getElementById(`label${i}`);
          // REMOVE PREVIOUS IMAGES
          const images_to_remove = document.getElementById(`img${i}`);
          if (images_to_remove) {
            label_assign.removeChild(images_to_remove);
          }
          // CREATE IMAGE
          let element = document.createElement("img");
          element.setAttribute("src", data[i]);
          element.setAttribute("style", `left: ${60 + column * 160}px; top: ${40 + row * 140}px;`);
          element.setAttribute("class", "image_frame");
          element.setAttribute("id", `img${i}`)
          label_assign.appendChild(element);
          column++;
        }
      }
    })
  } catch (e) {
    alert(`Error has occured ${e}`);
  }
}


async function submit_req() {
  let model = document.querySelector('input[name="model_radio"]:checked').id;
  let image_id = document.querySelector('input[name="image"]:checked').id;
  if (image_id == ""){
    alert("Select db")
    return 0
  }
  let label_number = image_id.replace( /^\D+/g, '');
  let label = document.getElementById(`label${label_number}`).childNodes;
  console.log(label[0]);
  const pred = label[0].src;
  console.log(typeof(pred), pred)
  if (model=="CNN") {
    try {
      $.ajax({
        type: 'POST',
        headers: {"Content-Type":"application/json;charset=UTF-8"},
        url: "/API/CNN",
        data: JSON.stringify({
          "image2predict":pred
        })
      })
      return 1
    } catch (e) {
      alert(`Error has occured ${e}`);
    }
  }
  if (model=="ResNet") {
    try {
      $.ajax({
        type: 'POST',
        headers: {"Content-Type":"application/json;charset=UTF-8"},
        url: "/API/ResNet",
        data: JSON.stringify({
          "image2predict":pred
        })
      })
      return 1
    } catch (e) {
      alert(`Error has occured ${e}`);
    }
  }
  if (model=="NN"){
    try {
      $.ajax({
        type: 'POST',
        headers: {"Content-Type":"application/json;charset=UTF-8"},
        url: "/API/NN",
        data: JSON.stringify({
          "image2predict":pred
        })
      })
      return 1
    } catch (e) {
      alert(`Error has occured ${e}`);
    }
  if (model=="LSTM"){
    try {
      $.ajax({
        type: 'POST',
        headers: {"Content-Type":"application/json;charset=UTF-8"},
        url: "/API/LSTM",
        data: JSON.stringify({
          "image2predict":pred
        })
      })
      return 1
    } catch (e) {
      alert(`Error has occured ${e}`);
    }
  }
  } 
  else {
    alert("No model was selected");
    return "Error";
  }

}
