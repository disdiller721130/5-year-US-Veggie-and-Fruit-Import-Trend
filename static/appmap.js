// Create a veggie map
function veggiemaps(veggie) {
  var url = `/veggieexp/${veggie}`;
  console.log(url);
  d3.json(url, function(veggiesummaries) {
    console.log(veggiesummaries);
    var vegshare = [];
    veggiesummaries.forEach(function(veggiesummary) {
      var color = "";
      if (veggiesummary.Share > 100) {
        color = "pink";
      }
      else if (veggiesummary.Share > 60) {
        color = "blue";
      }
      else if (veggiesummary.Share > 30) {
        color = "green";
      }
      else {
        color = "cyan";
      }
      var locations = [];
      locations.push(veggiesummary.lat , veggiesummary.lon);
      vegshare.push(
        L.circle(locations, {
          stroke: false,
          weight: 1,
          fillOpacity: 0.75,
          color: "white",
          fillColor: color,
          radius: veggiesummary.Share * 10000
        }).bindPopup("<h1>" + veggiesummary.Country + "</h1> <hr> <h3>Shares(%): " + veggiesummary.Share + "</h3>")
      );
    });
    var vegLayer = L.layerGroup(vegshare);
    console.log(vegLayer);

    var streetmap = L.tileLayer("https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}", {
    attribution: "Map data &copy; <a href=\"https://www.openstreetmap.org/\">OpenStreetMap</a> contributors, <a href=\"https://creativecommons.org/licenses/by-sa/2.0/\">CC-BY-SA</a>, Imagery © <a href=\"https://www.mapbox.com/\">Mapbox</a>",
    maxZoom: 18,
    id: "mapbox.streets",
    accessToken: "pk.eyJ1IjoiZGlzZGlsbGVyNzIxMTMwIiwiYSI6ImNqd2s3YTBmNjA1MzE0OW1xcGk0em5xYjIifQ.FOinrzjB3Rdkhus3tQoOuA"
    });

    var baseMaps = {
      "Street Map": streetmap
    };

    var overlayMaps = {
      "Country": vegLayer
    };

    var myMap = L.map("map", {
      center: [
        37.09, -95.71
      ],
      zoom: 1,
      layers: [streetmap, vegLayer]
    });

    L.control.layers(baseMaps, overlayMaps).addTo(myMap);
    console.log(myMap);
  });  
}

//Switch to Fruit Map
function fruitmaps(fruit) {
  var fruitUrl = `/fruitexp/${fruit}`;
  console.log(fruitUrl);
  d3.json(fruitUrl, function(fruitsummaries) {
    console.log(fruitsummaries);
    var fruitshare = [];

    fruitsummaries.forEach(function(fruitsummary) {
      var color = "";
      if (fruitsummary.Share > 100) {
        color = "pink";
      }
      else if (fruitsummary.Share > 60) {
        color = "blue";
      }
      else if (fruitsummary.Share > 30) {
        color = "green";
      }
      else {
        color = "cyan";
      }
      var locations = [];
      locations.push(fruitsummary.lat , fruitsummary.lon);
      fruitshare.push(
        L.circle(locations, {
          stroke: false,
          weight: 1,
          fillOpacity: 0.75,
          color: "white",
          fillColor: color,
          radius: fruitsummary.Share * 10000
        }).bindPopup("<h1>" + fruitsummary.Country + "</h1> <hr> <h3>Shares(%): " + fruitsummary.Share + "</h3>")
      );
    });
    var fruitLayer = L.layerGroup(fruitshare);
    console.log(fruitLayer);

    var streetmap = L.tileLayer("https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}", {
    attribution: "Map data &copy; <a href=\"https://www.openstreetmap.org/\">OpenStreetMap</a> contributors, <a href=\"https://creativecommons.org/licenses/by-sa/2.0/\">CC-BY-SA</a>, Imagery © <a href=\"https://www.mapbox.com/\">Mapbox</a>",
    maxZoom: 18,
    id: "mapbox.streets",
    accessToken: "pk.eyJ1IjoiZGlzZGlsbGVyNzIxMTMwIiwiYSI6ImNqd2s3YTBmNjA1MzE0OW1xcGk0em5xYjIifQ.FOinrzjB3Rdkhus3tQoOuA"
    });

    var baseMaps = {
      "Street Map": streetmap
    };

    var overlayMaps = {
      "Country": fruitLayer
    };

    var myMap = L.map("map", {
      center: [37.09, -95.71],
      zoom: 1,
      layers: [streetmap, fruitLayer]
    });

    L.control.layers(baseMaps, overlayMaps).addTo(myMap);
  });
  
}
  
//initiation
function mapinit() {
  // Grab a reference to the dropdown select element
  var productselector = d3.select("#productDataset");
  var typeselector = d3.select("#typeDataset");
  var productType = ["Veggie","Fruit"];
  productType.forEach(function(type) {
    typeselector.append("option").text(type).property("value", type)
  });
  // Use the list of sample names to populate the select options
  d3.json("/veggienames", function(vegNames) {
    vegNames.forEach((veg) => {
      productselector
        .append("option")
        .text(veg)
        .property("value", veg);
    });
    var firstVeg = vegNames[0];
    console.log(firstVeg);
    veggiemaps(firstVeg);

    d3.json(`/product/${firstVeg}`, function(item) {
      var productPanel = d3.select("#product_prediction");
      productPanel.html("");
      Object.entries(item).forEach(([key, value]) => {
        productPanel.append("h6").text(`${key}: ${value}`)
      });
    });
  });  
}
mapinit();

//Choose the type of products: Fruit or vegetable.
function getValue(value) {
  map.remove();
  var mapCreate = d3.select(".col-md-10");
  mapCreate.append("div").property("id","map");
  d3.select("#productDataset").html("");

  if (value == "Fruit") {
    var fruitSelect = d3.select("#productDataset");
    d3.json("/fruitnames", function(fruitName) {
      fruitName.forEach((fruit) => {
        fruitSelect.append("option").text(fruit).property("value", fruit);
      });
      var firstFruit = fruitName[0];
      console.log(firstFruit);
      fruitmaps(firstFruit);

      d3.json(`/product/${firstFruit}`, function(item) {
        var productPanel = d3.select("#product_prediction");
        productPanel.html("");
        Object.entries(item).forEach(([key, value]) => {
          productPanel.append("h6").text(`${key}: ${value}`)
        });
      });
    });
  }
  else {
    var vegSelect = d3.select("#productDataset");
    d3.json("/veggienames", function(vegetable) {
      vegetable.forEach((veg) => {
        vegSelect.append("option").text(veg).property("value", veg);
      });
      var secondVeg = vegetable[0];
      veggiemaps(secondVeg);

      d3.json(`/product/${secondVeg}`, function(item) {
        var productPanel = d3.select("#product_prediction");
        productPanel.html("");
        Object.entries(item).forEach(([key, value]) => {
          productPanel.append("h6").text(`${key}: ${value}`)
        });
      });
    });
  }
}

//Switch to the other fruits or vegetables
function optionChanged(NewOption) {
  map.remove();
  var mapCreate = d3.select(".col-md-10");
  mapCreate.append("div").property("id","map");
  d3.json("/veggienames", function(items) {
    console.log(items);
    var count = 0;
    items.forEach(function(item) {
      if(item == NewOption) {
        count += 1;
      }
    });
    if (count == 1) {
      veggiemaps(NewOption);
    }
    else {
      fruitmaps(NewOption);
    }
  });   

  d3.json(`/product/${NewOption}`, function(item) {
    var productPanel = d3.select("#product_prediction");
    productPanel.html("");
    Object.entries(item).forEach(([key, value]) => {
      productPanel.append("h6").text(`${key}: ${value}`)
    });
  });
}




