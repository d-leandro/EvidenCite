FROM nginx:alpine

COPY index.html /usr/share/nginx/html/index.html
COPY static /usr/share/nginx/html/static

COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80