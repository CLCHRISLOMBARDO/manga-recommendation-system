from flask import Flask, render_template, request, redirect, jsonify
import recomendador

app = Flask(__name__)

@app.get("/")
def get_index():
    return render_template("login.html")

@app.post("/")
def post_index():
    id_lector = request.form.get("id_lector")

    if not id_lector:
        return redirect("/") # render_template("login.html")

    # TODO: Validar que el lector existe en la base de datos
    return redirect(f"/recomendar/{id_lector}")


@app.get("/recomendar/<id_lector>")
def get_recomendar(id_lector):
    libros_recomendados, algoritmo = recomendador.recomendar(id_lector)

    cant_vistos = len(recomendador.items_vistos(id_lector))
    cant_valorados = len(recomendador.items_valorados(id_lector))

    return render_template("recomendaciones.html", cant_vistos=cant_vistos, cant_valorados=cant_valorados, libros_recomendados=libros_recomendados, id_lector=id_lector, algoritmo=algoritmo)


@app.post("/recomendar/<id_lector>")
def post_recomendar(id_lector):

    # guardar los items puntuados
    for id_libro in request.form.keys():
        rating = int(request.form.get(id_libro))
        if rating > 0: # 0 es que no lo puntuó
            recomendador.insertar_interaccion(id_lector, id_libro, rating)

    return redirect(f"/recomendar/{id_lector}")


@app.get("/reset/<id_lector>")
def get_reset(id_lector):
    recomendador.reset_usuario(id_lector)
    return redirect("/recomendar/" + id_lector)


### API

@app.get("/api/recomendar/<id_lector>")
def get_api_recomendar(id_lector):
    rec = recomendador.recomendar(id_lector)
    return jsonify({ "id_lector": id_lector, "recomendacion": rec })


